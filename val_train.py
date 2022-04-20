import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnnz
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from StixelsDataset import *
from utils.augmentations import SSDAugmentation, StixelAugmentation
from layers.modules import MultiBoxLoss, StixelLoss
from StixelNet import build_net
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import settings
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=2e-1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.9, type=float, help='Gamma update for SGD')
parser.add_argument('--basepath_d', type=str, help='The basepath of KittiTracking')
parser.add_argument('--basepath_s', type=str, help='The basepath of Kitti Raw Data')
parser.add_argument('--gt_path_s', type=str, help='The path of Stixel Ground Trurh')
parser.add_argument('--resume', type=str, help='The path of checkpoint')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
ssd_dim = (800, 370)  # the size of image after resize (width,height)
means = (104, 117, 123)
num_classes = 9 + 1
batch_size = args.batch_size


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if args.resume is None:
    net = build_net('train', ssd_dim, num_classes)
    vgg_weights = torch.load('weights/vgg16_reducedfc.pth')
    print('Loading base network...')
    net.vgg.load_state_dict(vgg_weights)
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
else:
    net = torch.load(args.resume)

savename = 'weights/kitti_%f_%.3f' % (args.lr, args.gamma)

cudnnz.benchmark = True

net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)


def stixel_train():


    net.train()
    printfrq = 1
    step = 0
    dataset = StixelsDataset(args.basepath_s, args.gt_path_s, StixelAugmentation(size=ssd_dim, mean=means))
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=False, pin_memory=not torch.cuda.is_available())

    validation_split = 0.1
    dataset_len = len(dataset)
    indices = list(range(dataset_len))
    val_len = int(np.floor(validation_split * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, sampler=validation_sampler)
    data_loaders = {"train": train_loader, "val": validation_loader}
    data_lengths = {"train": len(train_idx), "val": val_len}

    n_epochs = 40


    lossfunction = StixelLoss()
    minloss = 9999
    min_sum_loss = 9999

    for epoch in range(100):
        if epoch % 10 == 0:
            lr = adjust_learning_rate(optimizer, args.gamma, step)
            step = step + 1

        avgloss = 0
        current_sum_loss = 0
        for i, (images, havetargets, targets) in enumerate(data_loader):
            # np_arr = images.cpu().detach().numpy()[0]
            # #np_arr = np_arr.reshape(np_arr.shape[1:])
            # np_arr = np.transpose(np_arr, (1, 2, 0))
            # np_arr = np_arr.astype(int)
            # np_arr = np.ascontiguousarray(np_arr, dtype=np.uint8)
            # np_tars = targets.cpu().detach().numpy()[0].reshape((100))
            # he, wid = np_arr.shape[0], np_arr.shape[1]
            # for ind, stix in enumerate(np_tars):
            #     xcor = int(ind * wid / 100.)
            #     ycor = int(stix * he / 50)
            #     if ycor != 0:
            #         cv2.circle(np_arr, (xcor, ycor), 4, color=(0, 255, 0), thickness=1)
            # cv2.imshow('aa', np_arr)
            # cv2.waitKey(0)

            images = Variable(images).to(device)
            havetargets = Variable(havetargets).to(device)
            targets = Variable(targets).to(device)
            dec, stixel = net(images)
            optimizer.zero_grad()
            loss = lossfunction(stixel, havetargets, targets)
            loss.backward()
            optimizer.step()
            avgloss = avgloss + loss.data.item()
            current_sum_loss = current_sum_loss + loss.data.item()

        print("Epoch: %d batch: %d lr: %.6f loss: %.6f" % (epoch, i, lr, current_sum_loss))

        if current_sum_loss < min_sum_loss:
            min_sum_loss = current_sum_loss
            torch.save(net.state_dict(), savename + ('_%d_%.6f.pt' % (epoch, current_sum_loss)))


    torch.save(net.state_dict(), savename + ('_%d.pt' % epoch))


def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    stixel_train()
