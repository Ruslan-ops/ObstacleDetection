from RegNet import *
import cv2
import torch
from torch.autograd import Variable
import torchvision.transforms.functional

device = 'cuda' if torch.cuda.is_available() else 'cpu'

w0 = 24
wa = 36
wm = 2.5
#d_list = [1, 2, 7, 12]
b = 1
d = 13
g = 8
s = 2
se = 8


reg = RegNetX(initial_width=w0, slope=wa, quantized_param=wm, network_depth=d,bottleneck_ratio=b, group_width=g, stride=s, se_ratio=se)
reg.eval()

image = cv2.imread('../2011_09_26_46_0000000020.png')
image = torchvision.transforms.functional.to_tensor(image)
#image = image.permute(2, 0, 1)
image = image[np.newaxis, ...]
y = reg(image)
lin = y.contiguous().view(-1)
sum = lin.sum()
print(y, y.shape)