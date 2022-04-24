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
import numpy as np
import settings
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=2e-2, type=float, help='initial learning rate')
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
    #vgg_weights = torch.load('weights/vgg16_reducedfc.pth')
    #print('Loading base network...')
    #net.vgg.load_state_dict(vgg_weights)
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
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


def train_one_step(model, data, optimizer):
    optimizer.zero_grad()
    (images, havetargets, targets) = data
    images = Variable(images).to(device)
    havetargets = Variable(havetargets).to(device)
    targets = Variable(targets).to(device)
    _, stixel = model(images)
    loss = StixelLoss()(stixel, havetargets, targets)
    loss.backward()
    optimizer.step()
    return loss.data.item()

def train_one_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch_index,  data in enumerate(data_loader):
        loss = train_one_step(model, data, optimizer)
        total_loss += loss
    scheduler.step()
    return total_loss

def validate_one_step(model, data):
    (images, havetargets, targets) = data
    images = Variable(images).to(device)
    havetargets = Variable(havetargets).to(device)
    targets = Variable(targets).to(device)
    _, stixel = model(images)
    loss = StixelLoss()(stixel, havetargets, targets)
    return loss.data.item()

def validate_one_epoch(model, data_loader):
    model.eval()
    total_loss = 0
    for batch_index, data in enumerate(data_loader):
        with torch.no_grad():
            loss = validate_one_step(model, data)
        total_loss += loss
    return total_loss

def stixel_train():
    augmentation = StixelAugmentation(size=ssd_dim)
    dataset = StixelsDataset(args.basepath_s, augmentation.transform, augmentation.target_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_data_loader = data.DataLoader(train_dataset, batch_size, num_workers=args.num_workers,
                                      shuffle=True, pin_memory=not torch.cuda.is_available())
    val_data_loader = data.DataLoader(val_dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, pin_memory=not torch.cuda.is_available())



    min_val_loss = 9999
    print('train started')
    model = net

    log_path = 'log.txt'
    with open(log_path, 'w') as log:
        for epoch in range(100):
            total_train_loss = train_one_epoch(model, train_data_loader, optimizer, scheduler)
            total_val_loss = validate_one_epoch(model, val_data_loader)

            avg_train_loss = total_train_loss / len(train_data_loader)
            avg_val_loss = total_val_loss / len(val_data_loader)

            print("Epoch: %d lr: %.6f train_loss: %.6f val_loss: %.6f" % (epoch, optimizer.param_groups[0]['lr'], avg_train_loss, avg_val_loss))

            model_name = savename + ('_%d_%.6f.pt' % (epoch, avg_val_loss))
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                torch.save(net.state_dict(), model_name)

            info = f'{epoch}\t{avg_train_loss}\t{avg_val_loss}'
            log.write(info + '\n')



def stixel_train_OOOOld():
    net.train()
    printfrq = 1
    step = 0
    augmentation = StixelAugmentation(size=ssd_dim)
    dataset = StixelsDataset(args.basepath_s, augmentation.transform, augmentation.target_transform)
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, pin_memory=not torch.cuda.is_available())
    lossfunction = StixelLoss()
    minloss = 9999
    min_sum_loss = 9999
    print('train started')

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
            print(f'finished {i + 1}/{len(data_loader)}')

        current_sum_loss = current_sum_loss / (i + 1)
        print("Epoch: %d batch: %d lr: %.6f loss: %.6f" % (epoch, i, lr, current_sum_loss))

        model_name = savename + ('_%d_%.6f.pt' % (epoch, current_sum_loss))
        if current_sum_loss < min_sum_loss:
            min_sum_loss = current_sum_loss
            torch.save(net.state_dict(), model_name)

        log_path = 'log.txt'
        with open(log_path, 'w') as log:
            info = f'{epoch}\t{current_sum_loss}'
            log.write(info + '\n')


    torch.save(net.state_dict(), savename + ('_%d.pt' % epoch))


def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    stixel_train()
