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

parser = argparse.ArgumentParser()
parser.add_argument('--basepath',type=str,help='The basepath of KittiTracking')
#parser.add_argument('--outpath',default='./eval/',type=str,help='The path of output demo')
parser.add_argument('--model',type=str,help='The path of model to evaluate')
#parser.add_argument('--index',type=int,default=1,help='the index of KittiTracking video')

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
means = (104, 117, 123)
ssd_dim = (800,370)

# if not os.path.exists(args.outpath):
#     os.mkdir(args.outpath)

def stixel_test(dataset,model):
    num_images = len(dataset)
    for i in range(num_images):
        if i % 10 != 0:
            continue
        im, have_targ, targ = dataset.__getitem__(i)
        oimg = dataset.get_original_image(i)
        h, w, c = oimg.shape
        x = Variable(im.unsqueeze(0)).to(device)
        _,stixel = model(x)
        predict=stixel.data.cpu().numpy()
        predict = predict[0]
        predict=predict.argmax(1)
        for x,py in enumerate(predict):
            x0=int(x*w/settings.STIXEL_COLOMNS_AMOUNT)
            x1=int((x+1)*w/settings.STIXEL_COLOMNS_AMOUNT)
            x = (x0 + x1)// 2
            y=int((py+0.5) * h / settings.BINS_AMOUNT)
            cv2.circle(oimg,(x,y),3,(0,0,255),-1)
        #cv2.imwrite(os.path.join(args.outpath,'%d.png'%i),oimg)
        print("finish %d/%d"%(i,num_images))
        cv2.imshow('aaa', oimg)
        cv2.waitKey(0)






if __name__ == '__main__':
    # load net
    num_classes = 9 + 1 # +1 background
    net = build_net('test', ssd_dim, num_classes)
    net.load_state_dict(torch.load(args.model, map_location=torch.device(device)))
    # if os.path.exists(args.model):
    #     a = 0
    #model_weights = torch.load(args.model)
    #model = torch.load(args.model)
    #net.load_state_dict(args.model)

    #net=torch.load(args.model)
    net.eval()
    print('Finished loading model!')
    augmentation = StixelAugmentation(size=ssd_dim)
    dataset = StixelsDataset(args.basepath, augmentation.train_target_transform)
    net = net.to(device)
    cudnn.benchmark = torch.cuda.is_available()
    stixel_test(dataset,net)
