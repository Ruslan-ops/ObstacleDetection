from regnet.RegNet import RegNetY
import torch
from settings import *
import torch.nn as nn
import torchvision.models as models
from regnet.local.custom_regnet import regnet_y_800mf, regnet_y_200mf, regnet_y_32gf, regnet_y_400mf

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class StixelHead(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(StixelHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.stixel = nn.Sequential(
            # nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1),
            # nn.ReLU(),
            nn.Dropout(p=0.2),
            #nn.Linear(in_features=num_channels, out_features=STIXEL_COLOMNS_AMOUNT * num_classes)
            nn.Conv2d(in_channels=num_channels, out_channels=STIXEL_COLOMNS_AMOUNT * num_classes, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.pool(x)
        #x = x.flatten(start_dim=1)
        x = self.stixel(x)
        x = x.view(x.shape[0], STIXEL_COLOMNS_AMOUNT, POSITION_NEURONS_AMOUNT)# + TYPE_NEURONS_AMOUNT
        #x = x.view(x.shape[:3], -1)#.permute(0, 2, 1)
        x = self.softmax(x)
        summm = x[0][3]
        summm = summm.sum()
        return x

def create_stixel_model(backbone_weights_path=None):
    model = regnet_y_400mf(pretrained=True)

    model.eval()
    # if backbone_weights_path is not None:
    #     model.load_state_dict(torch.load(backbone_weights_path, map_location=torch.device(device)))
    in_channels = model.fc.in_features
    model.fc = StixelHead(num_channels=in_channels, num_classes=POSITION_NEURONS_AMOUNT)# + TYPE_NEURONS_AMOUNT
    return model
# def create_stixel_model(backbone_weights_path=None):
#     w0 = 24
#     wa = 36
#     wm = 2.5
#     b = 1
#     d = 13
#     g = 8
#     s = 2
#     se = 4
#     model = RegNetY(initial_width=w0, slope=wa, quantized_param=wm, network_depth=d, bottleneck_ratio=b, group_width=g,
#                   stride=s, se_ratio=se)
#     if backbone_weights_path is not None:
#         model.load_state_dict(torch.load(backbone_weights_path, map_location=torch.device(device)))
#     in_channels = model.net.head.in_channels
#     model.net.head = StixelHead(num_channels=in_channels, num_classes=50)
#     return model