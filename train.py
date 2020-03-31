import torch
import numpy as np
from utils import *
from model import *
import torch.utils.data as data
import argparse
import time
import datetime
from terminaltables import AsciiTable
from tqdm import tqdm


if __name__ =='__main__':
    argparser = argparse.ArgumentParser(description='train the model')
    argparser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    argparser.add_argument('--save',type=str, default='./weights',help='the default directory of saved weights')
    argparser.add_argument('--weight',type=str,help='the yolov3 weights',default='weights/yolov3.weights')
    argparser.add_argument('--config',type=str,help='path of config file', default='cfg/yolov3pro.cfg')
    argparser.add_argument("--class_name", type=str, default="img/class_name.txt", help="path to class label file")
    argparser.add_argument('--train_path', type=str, default='img/train.txt',help='directory of training images')
    argparser.add_argument('--valid_path', type=str, default='img/valid.txt', help='directory of validating images')
    argparser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    argparser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    argparser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    argparser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    argparser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    argparser.add_argument('--pretrained', type=bool, default=False, help='whether use a pretrained weightsfile')
    argparser.add_argument("--checkpoint_weights", type=str,default='weights/checkpoint/', help="path to checkpoint model")
    argparser.add_argument('--pretrained_weights', type=str, default='weights/pretrained.weights')
    argparser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = argparser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # creat a list containg the class names
    with open(opt.class_name,'r') as fd:
        class_names = fd.read().split('\n')[:-1]
    
    # create the net
    model = MyNet(opt.config)
    # set the model in training mode
    model.train()
    # initialize the parameters, applying weights_init_normal on every submodule and model.self
    model.apply(weights_init_normal)

    if opt.pretrained:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_weights(opt.pretrained_weights)
    ### 
    #dataloader
    dataset = ListDataset(opt.train_path, img_size=opt.img_size,augment=True,multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    optimizer = torch.optim.Adam(model.parameters())
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    targets = 0

    for epoch in range(opt.epochs):
        start_time = time.clock()
        print('--- epoch{0} begins ---'.format(epoch))
        for index, (_,inputs, targets) in enumerate(tqdm(dataloader,desc="Training objects")):
            loss=0
            targets.requires_grad_(False)
            loss,output = model(inputs,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('loss={}'.format(loss))
        end_time = time.clock()
        print('epoch{} time:{:.3f}'.format(epoch,(end_time-start_time)))
        if (epoch+1) % 50 == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=opt.valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=1,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            #add to the logger
            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")    
   
        # if epoch%10==0:
        #     torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)

    model.save_weights(opt.save, cutoff=-1)
