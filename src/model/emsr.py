##enhanced mobile super resolution

from model import common

import torch.nn as nn


def make_model(args, parent=False):
    return EMSR(args)

class EMSR(nn.Module):
    def __init__(self, args, conv=common.groups_conv):
        super(EMSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        groups = args.n_groups
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [common.default_conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.group_ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale,groups=groups,
            ) for _ in range(n_resblocks)
        ]
        m_body.append(common.default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.group_Upsampler(common.default_conv, scale, n_feats, act=False),
            common.default_conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

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

