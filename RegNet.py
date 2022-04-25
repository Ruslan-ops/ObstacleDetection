import numpy as np

NUM_BLOCKS = [i for i in range(1, 17)]
BLOCK_WIDTH = [i for i in range(1, 1025) if i % 8 == 0]
BOTTLENECK_RATIO = [1, 2, 4]
GROUP_WIDTH = [1, 2, 4, 8, 16, 32]
SE_RARIO = 4
TRAIN_IMAGE_SIZE = 224
TEST_IMAGE_SIZE = 256
NUM_CLASSES = 1000
EIGENVALUES = [[0.2175, 0.0188, 0.0045]]
EIGENVECTORS = [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]


import torch.nn as nn
from math import sqrt

class Head(nn.Module):  # From figure 3

    def __init__(self, num_channels, num_classes):
        super(Head, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Stem(nn.Module): # From figure 3

    def __init__(self, out_channels):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.rl(x)
        return x


class XBlock(nn.Module): # From figure 4
    def __init__(self, in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio=None):
        super(XBlock, self).__init__()
        inter_channels = out_channels // bottleneck_ratio
        groups = inter_channels // group_width

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )

        if se_ratio is not None:
            se_channels = in_channels // se_ratio
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Conv2d(inter_channels, se_channels, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(se_channels, inter_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.se = None

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None
        self.rl = nn.ReLU()

    def forward(self, x):
        x1 = self.conv_block_1(x)
        x1 = self.conv_block_2(x1)
        if self.se is not None:
            x1 = x1 * self.se(x1)
        x1 = self.conv_block_3(x1)
        if self.shortcut is not None:
            x2 = self.shortcut(x)
        else:
            x2 = x
        x = self.rl(x1 + x2)
        return x


class Stage(nn.Module): # From figure 3
    def __init__(self, num_blocks, in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio):
        super(Stage, self).__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module("block_0", XBlock(in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio))
        for i in range(1, num_blocks):
            self.blocks.add_module("block_{}".format(i),
                                   XBlock(out_channels, out_channels, bottleneck_ratio, group_width, 1, se_ratio))

    def forward(self, x):
        x = self.blocks(x)
        return x


class AnyNetX(nn.Module):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetX, self).__init__()
        # For each stage, at each layer, number of channels (block width / bottleneck ratio) must be divisible by group width
        for block_width, bottleneck_ratio, group_width in zip(ls_block_width, ls_bottleneck_ratio, ls_group_width):
            assert block_width % (bottleneck_ratio * group_width) == 0
        self.net = nn.Sequential()
        prev_block_width = 32
        self.net.add_module("stem", Stem(prev_block_width))

        for i, (num_blocks, block_width, bottleneck_ratio, group_width) in enumerate(zip(ls_num_blocks, ls_block_width,
                                                                                         ls_bottleneck_ratio,
                                                                                         ls_group_width)):
            self.net.add_module("stage_{}".format(i),
                                Stage(num_blocks, prev_block_width, block_width, bottleneck_ratio, group_width, stride, se_ratio))
            prev_block_width = block_width
        self.net.add_module("head", Head(ls_block_width[-1], NUM_CLASSES))
        self.initialize_weight()

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.net(x)
        return x


class AnyNetXb(AnyNetX):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetXb, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio)
        assert len(set(ls_bottleneck_ratio)) == 1


class AnyNetXc(AnyNetXb):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetXc, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio)
        assert len(set(ls_group_width)) == 1


class AnyNetXd(AnyNetXc):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetXd, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio)
        assert all(i <= j for i, j in zip(ls_block_width, ls_block_width[1:])) is True


class AnyNetXe(AnyNetXd):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetXe, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio)
        if len(ls_num_blocks > 2):
            assert all(i <= j for i, j in zip(ls_num_blocks[:-2], ls_num_blocks[1:-1])) is True

class RegNetX(AnyNetXe):
    def __init__(self, initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride,
                 se_ratio=None):
        # We need to derive block width and number of blocks from initial parameters.
        parameterized_width = initial_width + slope * np.arange(network_depth)  # From equation 2
        parameterized_block = np.log(parameterized_width / initial_width) / np.log(quantized_param)  # From equation 3
        parameterized_block = np.round(parameterized_block)
        quantized_width = initial_width * np.power(quantized_param, parameterized_block)
        # We need to convert quantized_width to make sure that it is divisible by 8
        quantized_width = 8 * np.round(quantized_width / 8)
        ls_block_width, ls_num_blocks = np.unique(quantized_width.astype(np.int), return_counts=True)
        # At this points, for each stage, the above-calculated block width could be incompatible to group width
        # due to bottleneck ratio. Hence, we need to adjust the formers.
        # Group width could be swapped to number of groups, since their multiplication is block width
        ls_group_width = np.array([min(group_width, block_width // bottleneck_ratio) for block_width in ls_block_width])
        ls_block_width = np.round(ls_block_width // bottleneck_ratio / group_width) * group_width
        ls_group_width = ls_group_width.astype(np.int) * bottleneck_ratio
        ls_bottleneck_ratio = [bottleneck_ratio for _ in range(len(ls_block_width))]
        # print (ls_num_blocks)
        # print (ls_block_width)
        # print (ls_bottleneck_ratio)
        # print (ls_group_width)
        super(RegNetX, self).__init__(ls_num_blocks, ls_block_width.astype(np.int).tolist(), ls_bottleneck_ratio,
                                       ls_group_width.tolist(), stride, se_ratio)


class RegNetY(RegNetX):
    # RegNetY = RegNetX + SE
    def __init__(self, initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride,
                 se_ratio):
        super(RegNetY, self).__init__(initial_width, slope, quantized_param, network_depth, bottleneck_ratio,
                                      group_width, stride, se_ratio)