"""shufflenetv2 in pytorch



[1] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun

    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)
    
def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels / groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x

class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleUnit, self).__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )
        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True) 
            )

    
    def forward(self, x):

        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x
        
        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 3)
        
        return x


class conv_bn_relu(nn.Module):
    """docstring for conv_bn_relu"""

    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        out = self.norm(self.conv(x))
        if self.activation:
            out = F.relu(out, inplace=True)
        return out


def add_extras(i, batch_norm=False):
    layers = []
    in_channels = i
    channels = [128, 256, 128, 256, 128, 256]
    stride = [2, 2, 1, 1, 1, 1]
    padding = [1, 1, 0, 0, 0, 0]

    for k, v in enumerate(channels):
        if k % 2 == 0:
            if batch_norm:
                if k == 2:
                    layers += [conv_bn_relu(in_channels, v,
                                            kernel_size=4, padding=padding[k], stride = stride[k])]
                else:
                    layers += [conv_bn_relu(in_channels, v,
                                            kernel_size=3, padding=padding[k], stride = stride[k])]
            else:
                layers += [conv_relu(in_channels, v,
                                     kernel_size=1, padding=padding[k])]
        else:
            if batch_norm:
                layers += [conv_bn_relu(in_channels, v,
                                        kernel_size=3, stride=stride[k], padding=padding[k])]
            else:
                layers += [conv_relu(in_channels, v,
                                     kernel_size=3, stride=stride[k], padding=padding[k])]
        in_channels = v

    return layers


class conv_relu(nn.Module):
    """docstring for conv_relu"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              bias=False, **kwargs)

    def forward(self, x):
        out = F.relu(self.conv(x), inplace=True)
        return out



class ShuffleNetV2(nn.Module):

    def __init__(self,phase, size, cfg=None):
        ratio = 1
        super(ShuffleNetV2, self).__init__()
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            ValueError('unsupported ratio number')
        
        self.num_classes = 21
        self.phase = phase
        self.size = size
        self.cfg = cfg

        self.features = nn.Sequential()

        pre = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24)
        )

        self.features.add_module("pre", pre)

        stage2 = self._make_stage(24, out_channels[0], 3)
        self.features.add_module("stage2", stage2)
        
        stage3 = self._make_stage(out_channels[0], out_channels[1], 7)
        self.features.add_module("stage3", stage3)
        
        stage4 = self._make_stage(out_channels[1], out_channels[2], 3)
        self.features.add_module("stage4", stage4)

        conv5 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True) 
        )
        self.features.add_module("conv5", conv5)

        extras = add_extras(out_channels[3], batch_norm=True)
        self.extras = nn.ModuleList(extras)

        nchannels = [128, 256, 256, 128, 256]

        resblock = add_resblock(nchannels)

        self.resblock = nn.ModuleList(resblock)

        self.loc = nn.ModuleList()
        self.conf = nn.ModuleList()

        for i, x in enumerate([256] * 5):
            n = cfg.anchor_config.anchor_nums[i]
            self.loc.append(nn.Conv2d(x, n * 4, kernel_size=1))
            self.conf.append(nn.Conv2d(x, n * self.num_classes, kernel_size=1))

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

        # Global pooling:
        # Undefined as PyTorch's functional API can be used for on-the-fly
        # shape inference if input size is not ImageNet's 224x224

        # Fully-connected classification layer

        # num_inputs = self.stage_out_channels[-1]
        # self.fc = nn.Linear(num_inputs, self.num_classes)
        # self.init_params()

        # self.fc = nn.Linear(out_channels[3], class_num)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()
        for k, feat in enumerate(self.features):
            x = feat(x)
            # if k > 2:
            #     sources += [x]
        #sources += [x]
        print("input:", np.shape(x))
        for k, v in enumerate(self.extras):
            x = v(x)
            print("extra:", np.shape(x))
            if k != 2:
                sources += [x]

        for k, x in enumerate(sources):
            print("source", k, ":", np.shape(x))
            sources[k] = self.resblock[k](x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test': 
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.cfg.num_classes))  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.cfg.num_classes)
            )
        return output

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, 2))

        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
            repeat -= 1
        
        return nn.Sequential(*layers)

def add_resblock(nchannels):
    layers = []
    for k, v in enumerate(nchannels):
        layers += [ResBlock(v)]
    return layers

class ResBlock(nn.Module):
    """docstring for ResBlock"""

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.res1a = conv_relu(in_channels, 128, kernel_size=1)
        self.res1b = conv_relu(128, 128, kernel_size=3, padding=1)
        self.res1c = conv_relu(128, 256, kernel_size=1)

        self.res2a = conv_relu(in_channels, 256, kernel_size=1)

    def forward(self, x):
        out1 = self.res1a(x)
        out1 = self.res1b(out1)
        out1 = self.res1c(out1)

        out2 = self.res2a(x)
        out = out1 + out2
        return out

def shufflenetv2():
    return ShuffleNetV2()





