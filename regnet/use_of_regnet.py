from RegNet import *
import cv2
import torch
from torch.autograd import Variable
import torchvision.transforms.functional

device = 'cuda' if torch.cuda.is_available() else 'cpu'

w0 = 24
wa = 24
wm = 2.5
d_list = [1, 2, 7, 12]
b = 1
d = 4
g = 16
s = 2

reg = RegNetX(initial_width=w0, slope=wa, quantized_param=wm, network_depth=d,bottleneck_ratio=b, group_width=g, stride=s, se_ratio=None)
reg.eval()

image = cv2.imread('2011_09_26_46_0000000020.png')
image = torchvision.transforms.functional.to_tensor(image)
image = image.permute(2, 0, 1)
y = reg(image)
print(y)