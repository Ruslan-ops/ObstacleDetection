from regnet.RegNet import RegNetY, StixelHead
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_stixel_model(backbone_weights_path=None):
    w0 = 24
    wa = 36
    wm = 2.5
    b = 1
    d = 13
    g = 8
    s = 2
    se = 4
    model = RegNetY(initial_width=w0, slope=wa, quantized_param=wm, network_depth=d, bottleneck_ratio=b, group_width=g,
                  stride=s, se_ratio=se)
    if backbone_weights_path is not None:
        model.load_state_dict(torch.load(backbone_weights_path, map_location=torch.device(device)))
    in_channels = model.net.head.in_channels
    model.net.head = StixelHead(num_channels=in_channels, num_classes=50)
    return model