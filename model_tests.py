import torchvision.models as models
import cv2
import torchvision.transforms.functional
import numpy as np

regnet_y_800mf = models.regnet_y_800mf(pretrained=True)


regnet_y_800mf.eval()

image = cv2.imread('2011_09_26_46_0000000020.png')
image = torchvision.transforms.functional.to_tensor(image)
#image = image.permute(2, 0, 1)
image = image[np.newaxis, ...]
y = regnet_y_800mf(image)
print(y)