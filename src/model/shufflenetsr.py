##enhanced mobile super resolution

from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return SHUFFLENETSR(args)

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        # 维度变换之后必须要使用.contiguous()使得张量在内存连续之后才能调用view函数
        return x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

class Bottleneck(nn.Module):
    def __init__(self, n_feats,stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        # bottleneck层中间层的channel数变为输出channel数的1/4
        self.conv1 = nn.Conv2d(n_feats, n_feats,
                               kernel_size=1, groups=groups, bias=True)
        self.shuffle1 = ShuffleBlock(groups=groups)
        self.conv2 = nn.Conv2d(n_feats, n_feats,
                               kernel_size=3, stride=stride, padding=(3//2),
                               bias=True)        
        self.conv3 = nn.Conv2d(n_feats, n_feats,
                               kernel_size=3, stride=stride, padding=(3//2),
                               groups=n_feats, bias=True)
        self.conv4 = nn.Conv2d(n_feats, n_feats,
                               kernel_size=1, groups=groups, bias=True)
        self.BN = nn.BatchNorm2d(n_feats)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.BN(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.BN(self.conv3(out)))
        out = F.relu(self.BN(self.conv4(out)))
        res = self.shortcut(x)
        out = F.relu(out+res)
        return out

class SHUFFLENETSR(nn.Module):
    def __init__(self, args,conv=common.default_conv):
        super(SHUFFLENETSR, self).__init__()
        n_feats = args.n_feats
        n_bottlenecks = args.n_resblocks
        groups = args.n_groups
        scale = args.scale[0]
        kernel_size = 3 

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [conv(args.n_colors, n_feats, kernel_size)]  
        self.body = self._make_layer(n_feats, n_bottlenecks, groups)
        # define tail module
        m_tail = [
            conv(n_feats, n_feats, kernel_size),   
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)
    def _make_layer(self, n_feats, num_blocks, groups):
        layers = []
        for _ in range(num_blocks):
            layers.append(Bottleneck(n_feats,stride=1, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

