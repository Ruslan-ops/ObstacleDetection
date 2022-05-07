from __future__ import print_function
# import torch
# import torch.nn as nn
import os

import torch.backends.cudnn as cudnn
# import torchvision.transforms as transforms
from torch.autograd import Variable
from StixelsDataset import *
from utils.augmentations import StixelAugmentation
# import torch.utils.data as data
# import os
import argparse
import settings
from StixelNet import *

import numpy as np
import cv2
from StixelRegNet import *

parser = argparse.ArgumentParser()
parser.add_argument('--basepath',type=str,help='The basepath of KittiTracking')
#parser.add_argument('--outpath',default='./eval/',type=str,help='The path of output demo')
parser.add_argument('--model',type=str,help='The path of model to evaluate')
#parser.add_argument('--index',type=int,default=1,help='the index of KittiTracking video')

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
means = (104, 117, 123)
ssd_dim = (400,185)

# if not os.path.exists(args.outpath):
#     os.mkdir(args.outpath)

def stixel_test(dataset,model):
    num_images = len(dataset)
    for i in range(num_images):
        if i % 1 != 0:
            continue
        im, have_targ, targ = dataset.__getitem__(i)
        oimg = dataset.get_original_image(i)
        h, w, c = oimg.shape
        x = Variable(im.unsqueeze(0)).to(device)
        stixel = model(x)
        predict=stixel.data.cpu().numpy()
        predict = predict[0]
        predict=predict.argmax(1)
        for i,py in enumerate(predict):
            x = i
            x0=int(x*w/settings.STIXEL_COLOMNS_AMOUNT)
            x1=int((x+1)*w/settings.STIXEL_COLOMNS_AMOUNT)
            x = (x0 + x1)// 2
            #y=int((py+0.5) * h / settings.BINS_AMOUNT)
            y = (py+0.5) / settings.POSITION_NEURONS_AMOUNT
            y=int(h * (y * settings.MAX_STIXEL_HEIGHT_PART + (1 - settings.MAX_STIXEL_HEIGHT_PART)))
            targ[i] = targ[i] / settings.POSITION_NEURONS_AMOUNT
            #target_y = int((targ[i]+0.5) * h / settings.BINS_AMOUNT)
            target_y = int(h * (targ[i] * settings.MAX_STIXEL_HEIGHT_PART + (1 - settings.MAX_STIXEL_HEIGHT_PART)))
            cv2.circle(oimg,(x,y),3,(0,0,255),-1)
            cv2.circle(oimg,(x,target_y),3,(0,255,0),-1)
        #cv2.imwrite(os.path.join(args.outpath,'%d.png'%i),oimg)
        print("finish %d/%d"%(i,num_images))
        cv2.imshow('aaa', oimg)
        cv2.waitKey(0)






if __name__ == '__main__':
    # load net
    num_classes = 9 + 1 # +1 background
    net = create_stixel_model()
    net.load_state_dict(torch.load('weights/kitti_91_0.222033.pt', map_location=torch.device(device)))
    # if os.path.exists(args.model):
    #     a = 0
    #model_weights = torch.load(args.model)
    #model = torch.load(args.model)
    #net.load_state_dict(args.model)

    #net=torch.load(args.model)
    net.eval()
    print('Finished loading model!')
    augmentation = StixelAugmentation(size=ssd_dim)

    dataset = StixelsDataset('dataset/annotations.txt', train_target_transform=augmentation.train_target_transform,
                   val_target_transform=augmentation.val_target_transform)
    net = net.to(device)
    cudnn.benchmark = torch.cuda.is_available()
    stixel_test(dataset,net)
