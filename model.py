import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.autograd import Variable
from utils import *
import time

def read_config(cfgpath):
    '''
    Description:
    Open and read the configuration of the neural network  
    Args:
    cfgpath:the path of the configuration file
    Returns:
    A list of blocks, each of them is stored as a dictionary
    '''

    fd= open(cfgpath,'r')
    # read by row
    rows = fd.read().split('\n')
    # remove the empty lines                        
    rows = [x for x in rows if len(x) > 0]               
    # remove the comment
    rows = [x for x in rows if x[0] != '#']
    # remove the whitespaces              
    rows = [x.rstrip().lstrip() for x in rows]
    '''
    exp: rows = ['[convolutional]', 'batch_normalize=1', 'filters=32', 'size=3',...]
    '''
    list_of_blocks = []
    block = {}

    for line in rows:
        if line[0] == '[':
            if block != {}: # if not empty
                list_of_blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip().lstrip()
            continue
        key = line.split('=')[0].rstrip().lstrip()  # exp: ['size ',' 3']
        value = line.split('=')[1].rstrip().lstrip()
        block[key] = value # {'size':'3'}
    list_of_blocks.append(block) # append the last blok
    
    return list_of_blocks

# shortcut layer and route layer
class ShortcutLayer(nn.Module):
    def __init__(self, source, activation):
        super(ShortcutLayer, self).__init__()
        self.source_ = source
        self.activation_ = activation

class RouteLayer(nn.Module):
    def __init__(self, layers, filters):
        super(RouteLayer, self).__init__()        
        self.layers_ = layers
        self.filters_ = filters

# yolo layer
class YoloLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        '''
        create the YoloLayer

        '''
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size



    def compute_grid_offsets(self, grid_size, cuda=False):
        # calculate the length of each grid
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        # the origin unit of anchor is pixel, which should be divided by the len of grid
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
    
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2) # X = (batch_size, channels, grid, grid)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x [num_samples, num_anchors, g, g ]
        y = torch.sigmoid(prediction[..., 1])  # Center y 
        w = prediction[..., 2]  # Width [num_samples, num_anchors, g, g ]
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Confidence
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Class pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape) # [num_samples, num_anchors, g, g, 4 ]
        pred_boxes[..., 0] = x.data + self.grid_x # [num_samples, num_anchors, g, g ] * [1,1,g,g]
        pred_boxes[..., 1] = y.data + self.grid_y # [num_samples, num_anchors, g, g ] * [1,1,g,g]
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w # [num_samples, num_anchors, g, g ] * [1,num_anchors,1,1]
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h # [num_samples, num_anchors, g, g ] * [1,num_anchors,1,1]

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride, # [num_samples, num_anchors*g*g, 4]
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            dim=-1,
        ) # [num_samples, num_anchors*g*g, 4+1+num_classes]

        if targets is None: # for perdiction, no need to calculate the loss
            return output, 0
        else: # for training
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            obj_mask = obj_mask.bool()
            noobj_mask = noobj_mask.bool()
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean() # class accuracy
            conf_obj = pred_conf[obj_mask].mean() 
            conf_noobj = pred_conf[noobj_mask].mean() 
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)


            # TODO cuda.to_cpu()

            self.metrics = {
                "loss":  to_cpu(total_loss).item(),
                "loss_x":  to_cpu(loss_x).item(),
                "loss_y":  to_cpu(loss_y).item(),
                "loss_w":  to_cpu(loss_w).item(),
                "loss_h":  to_cpu(loss_h).item(),
                "loss_conf":  to_cpu(loss_conf).item(),
                "loss_cls":  to_cpu(loss_cls).item(),
                "cls_acc":  to_cpu(cls_acc).item(),
                "recall50":  to_cpu(recall50).item(),
                "recall75":  to_cpu(recall75).item(),
                "precision":  to_cpu(precision).item(),
                "conf_obj":  to_cpu(conf_obj).item(),
                "conf_noobj":  to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss
    
def create_modules(list_of_blocks):
    '''
    Create the backbone network
    Args:
    The list of dictionaries containing the configuration of each block
    Return:
    A ModuleList, and a normal list containng input information
    '''
    net_info = list_of_blocks[0]
    sequential_list = nn.ModuleList()

    eps = float(net_info['exposure'])
    momentum = float(net_info['momentum'])
    decay = float(net_info['decay'])
    angle = float(net_info['angle'])
    input_filters = int(net_info['channels'])
    saturation = float(net_info['saturation'])
    hue = float(net_info['hue'])

    list_of_output_filters = []
    list_of_types=[]

    for index,block in enumerate(list_of_blocks[1:]):
        sequential = nn.Sequential() # re-init the sequential
       
        
        if block['type'] == 'convolutional':
            activation = block['activation']
            output_filters = int(block['filters'])
            kernel_size = int(block['size'])
            pad = int(block['pad'])
            stride = int(block['stride'])
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            if pad:
                # to keep the size of image only have two options: same or shrink to halfsize
                pad = (kernel_size - 1) // 2 # 1 ? kernel_size==3 : 0
            
            # append modules to the sequential
            conv2d = nn.Conv2d(input_filters,output_filters,kernel_size,stride,padding=pad,bias=bias)
            sequential.add_module('conv_{0}'.format(index), conv2d)

            if batch_normalize:
                bn = nn.BatchNorm2d(output_filters,eps=eps,momentum=momentum)
                sequential.add_module('batch_norm_{0}'.format(index), bn)

            if activation =='leaky':
                relu = nn.LeakyReLU(0.1, inplace = True)
                sequential.add_module('leaky_{0}'.format(index), relu)

        elif block['type'] == 'upsample':
            factor = block['stride']
            up = nn.Upsample(scale_factor=factor, mode='nearest')
            sequential.add_module('upsample_{0}'.format(index), up)
            output_filters = input_filters

        elif block['type'] == 'shortcut':
            from_=int(block['from'])
            shortcut = ShortcutLayer(from_,block['activation'])
            sequential.add_module('shortcut_{0}'.format(index), shortcut)
            output_filters = input_filters

        elif block['type'] == 'route':
            layers = block['layers'].split(',') # exp {'layers':'-1, 36'}
            layers = [x.rstrip().lstrip() for x in layers] # ['-1 ', ' 66']
            layers = [int(x) for x in layers] # [-1, 36]
            for idx,item in enumerate(layers):
                if item<0:
                    layers[idx] = index+item
            output_filters = sum([list_of_output_filters[i] for i in layers])
            route = RouteLayer(layers,output_filters)
            sequential.add_module('route_{0}'.format(index),route)
        
        elif block['type'] == 'yolo':
            anchors = block['anchors'].split(',')
            anchors = [x.rstrip().lstrip() for x in anchors]
            anchors = [int(x) for x in anchors]
            mask = block['mask'].split(',')
            mask = [x.rstrip().lstrip() for x in mask]
            mask = [int(x) for x in mask]
            # mask =[0,1,2] anchors = [10,13,16,30, ....]
            anchors_ = [(anchors[mask[0]*2],anchors[mask[0]*2+1]),
                        (anchors[mask[1]*2],anchors[mask[1]*2+1]),
                        (anchors[mask[2]*2],anchors[mask[2]*2+1]),]
            num_classes = int(block['classes'])
            input_dim = int(net_info['height'])
            
            yolo = YoloLayer(anchors_, num_classes, input_dim)
            sequential.add_module('yolo_{0}'.format(index), yolo)
        
        input_filters = output_filters
        list_of_output_filters.append(output_filters)
        list_of_types.append(block['type'])
        sequential_list.append(sequential) 

    return sequential_list, list_of_types, net_info

class MyNet(nn.Module):
    def __init__(self,config_path):
        super(MyNet,self).__init__()
        self.cfgfile = config_path
        self.list_of_config = read_config(self.cfgfile)
        self.seqs, self.list_of_types, self.net_info = create_modules(self.list_of_config)
        self.yolo_layers = [layer[0] for layer in self.seqs if hasattr(layer[0], "metrics")]
        self.seen = 0
        self.header = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self,x, targets=None):
        outputs = []
        total_loss = 0
        yolo_outputs = []

        for index,seq in enumerate(self.seqs):
            seq_type = self.list_of_types[index]
            if seq_type == 'convolutional' or seq_type == 'upsample':
                x = seq(x)
            elif seq_type == 'shortcut':
                from_ = seq[0].source_
                x = outputs[index-1] + outputs[index+from_]
            elif seq_type == 'route':
                layers = seq[0].layers_
                inputs = [outputs[l] for l in layers]
                x = torch.cat(inputs,dim=1) # inputsize = (Batchsize,Channels,Height,Width)
            elif seq_type == 'yolo':
                pred, loss = seq[0](x,targets=targets, img_dim=int(self.net_info['height']) )
                total_loss+=loss
                yolo_outputs.append(pred) # [num_samples, all_anchors, 4+1+80]
            outputs.append(x)
        yolo_outputs = torch.cat(yolo_outputs, 1)
        return   yolo_outputs if targets==None else  (total_loss,yolo_outputs)

    def load_weights(self,weights_path,cutoff=None):
        '''
        open and parse weights_file
        '''
        with open(weights_path,'rb') as fd:
            header = np.fromfile(fd, dtype=np.int32, count=5) # get the header for first 5 lines
            self.header = header
            self.seen = header[3]
            weights = np.fromfile(fd, dtype=np.float32) # read the rest file
        
        # Establish cutoff for loading backbone weights
        cutoff = cutoff
        if "darknet53.conv.74" in weights_path:
            cutoff = 75
        ptr = 0
        for index, (block, seq) in enumerate(zip(self.list_of_config[1:],self.seqs)):
            if index == cutoff:
                break            
            if block['type'] == 'convolutional':
                conv_layer = seq[0]
                try:
                    block['batch_normalize']
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = seq[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr+=num_b
                except:
                    #Load conv.bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                #load conv weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        with open(path+'/'+time.strftime("%Y%m%d-%H%M%S", time.localtime())+'.weights', "wb") as fd:
            self.header[3] = self.seen
            self.header.tofile(fd)
        # np.array.tofile()
        # bn.bias-->bn.weights-->bn.running_mean-->bn.running_var-->conv.weights
        # conv.bias-->conv.weights
            # Iterate through layers
            for i, (module_def, module) in enumerate(zip(self.list_of_config[1:cutoff], self.seqs[:cutoff])):
                if module_def["type"] == "convolutional":
                    conv_layer = module[0]
                    # If batch norm, load bn first
                    try:
                        module_def["batch_normalize"]
                        bn_layer = module[1]
                        bn_layer.bias.data.cpu().numpy().tofile(fd)
                        bn_layer.weight.data.cpu().numpy().tofile(fd)
                        bn_layer.running_mean.data.cpu().numpy().tofile(fd)
                        bn_layer.running_var.data.cpu().numpy().tofile(fd)
                    # Load conv bias
                    except:
                        conv_layer.bias.data.cpu().numpy().tofile(fd)
                    # Load conv weights
                    conv_layer.weight.data.cpu().numpy().tofile(fd)

           
            

        

if __name__ == '__main__':
    cfgfile = 'cfg/yolov3.cfg'
    listone = read_config(cfgfile)
    seqs,listoftypes,info = create_modules(listone)
    print(info)




