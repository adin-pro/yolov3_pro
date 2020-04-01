# convert the labels of imgs in kitti to yolo-style labels

import os
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str,help='original labels directory',default='/mnt/d/dataset/training/label_2' )
parser.add_argument('--output',type=str,help='transformed labels directory',default='yolo-labels')
parser.add_argument('--classes',type=str,help='path of class_names.txt',default='/mnt/d/dataset/training/class.name')
opt = parser.parse_args()

def transform(size_dict_list,class_names,img_size=(1242.0,375.0)):
    '''
    transform the coordinates
    output:[...
            [filename,class_num,x,y,w,h],
            ...]
    '''
    list_trans=[]
    for size_dict in size_dict_list:
        filename = size_dict['filename']
        class_type = size_dict['type']
        class_num = class_names[class_type]
        left = size_dict['left']
        top = size_dict['top']
        right = size_dict['right']
        bottom = size_dict['bottom']
        x = (left+right)/2.0/float(img_size[0])
        y = (top+bottom)/2.0/float(img_size[1])
        w = abs(left-right)/float(img_size[0])
        h = abs(top-bottom)/float(img_size[1])
        dict_trans = [filename,class_num,x,y,w,h]
        list_trans.append(dict_trans)
    return list_trans

if __name__ == '__main__':
    filename_list = os.listdir(opt.input)
    list_objects=[]
    
    for filename in filename_list:
        with open(opt.input+'/'+filename,'r') as fd:
            labels = fd.read().split('\n')
            labels = [i.split(' ') for i in labels]
            labels = [i for i in labels if i!=['']]
            for label in labels:
                if label[0]=='DontCare':
                    continue
                object_info={}
                object_info['filename'] = filename
                object_info['type']  = label[0]
                object_info['left']  = float(label[4])
                object_info['top']   = float(label[5])
                object_info['right'] = float(label[6])
                object_info['bottom']= float(label[7])
                list_objects.append(copy.deepcopy(object_info))

    with open(opt.classes,'r') as fd:
        class_list = fd.read().split('\n')
        class_dict = {}
        for index,name in enumerate(class_list):
            class_dict[name]=index

    objects_trans = transform(list_objects,class_dict)
    
    for object_ in objects_trans:
        filename = object_[0]
        with open(opt.output+'/'+filename,'a+') as fd:
            fd.write(str(object_[1])+' '
                    +str(object_[2])+' '
                    +str(object_[3])+' '
                    +str(object_[4])+' '
                    +str(object_[5])+'\n')
    print('output directory:{}'.format(opt.output))
    print('{} labels finished'.format(len(objects_trans)))

