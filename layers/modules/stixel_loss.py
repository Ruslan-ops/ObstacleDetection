import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import pickle
import math


class StixelLoss(nn.Module):

    def __init__(self):
        super(StixelLoss, self).__init__()
        self.epoch=0

    def forward(self, predect, havetarget, target):
        loss = self._PL_loss(predect, havetarget, target)
        return loss

    def _PL_loss(self, predect, havetarget, target):
        target = (target - 0.5).view(target.size(0), target.size(1), 1)
        target = (target - torch.floor(target)) + torch.floor(target) + 1e-12
        target = target.view(target.size(0), target.size(1), 1)
        f_tensor = torch.floor(target).type(torch.LongTensor)
        c_tensor = torch.ceil(target).type(torch.LongTensor)
        if torch.cuda.is_available():
            f_tensor = f_tensor.type(torch.cuda.LongTensor)
            c_tensor = c_tensor.type(torch.cuda.LongTensor)
        fp = torch.gather(predect, 2, f_tensor)
        cp = torch.gather(predect, 2, c_tensor)
        p = fp * (torch.ceil(target) - target) + cp * (target - torch.floor(target)) + 1e-12
        p = p.view(havetarget.size(0), havetarget.size(1))
        loss = -torch.log(p) * havetarget
        loss = torch.sum(loss) / torch.sum(havetarget)
        if math.isnan(loss.data.item()):
            a = 0
        return loss