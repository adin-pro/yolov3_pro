import torch
import numpy as np
import argparse
import torch.utils.data as data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from model import *
from utils import *
import time
import datetime

if __name__ =='__main__':
    argparser = argparse.ArgumentParser(description='detect the image')
    argparser.add_argument('-src','--source', type=str,help='the source directory', default='img/test')
    argparser.add_argument('-des','--destination',type=str, help='the directory containg the output image', 
                            default='img/test_out')
    argparser.add_argument('-wght','--weight',type=str,help='the yolov3 weights',default='weights/yolov3pro.weights')
    argparser.add_argument('-cfg','--config',type=str,help='path of config file', default='cfg/yolov3pro.cfg')
    argparser.add_argument('-clsn',"--class_name", type=str, default="img/class_name.txt", help="path to class label file")
    argparser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    argparser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    argparser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    argparser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    argparser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    argparser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = argparser.parse_args()

    SOURCE_PATH = opt.source
    DEST_PATH = opt.destination
    WEIGHTS_FILE = opt.weight
    CONFIG_PATH = opt.config
    CLASS_NAME  = opt.class_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DEST_PATH, exist_ok=True)
    # create the net
    model = MyNet(CONFIG_PATH)
    # load the weight
    model.load_weights(WEIGHTS_FILE)
    # set the module in evaluation mode
    model.eval()

    with open(CLASS_NAME,'r') as fd:
        class_names = fd.read().split('\n')[:-1]


    dataloader = data.DataLoader(
        ImageFolder(SOURCE_PATH, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()

    # start perdiction
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = input_imgs.type(Tensor)

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            # print(type(detections))
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (class_names[int(cls_pred)], cls_conf.item()))

                box_w = 3*(x2 - x1)
                box_h = 3*(y2 - y1)

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=class_names[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(opt.destination+ f"/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()








