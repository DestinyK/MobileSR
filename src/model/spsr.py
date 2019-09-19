from model import common

import torch.nn as nn


def make_model(args, parent=False):
    return SPSR(args)

class SPSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SPSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        print(n_resblocks,n_feats)
        kernel_size = 3
        act = nn.ReLU(True)
        scale = args.scale[0]
        self.scale_idx = 0
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        self.pre_process = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv, n_feats, 5, act=act),
            ) for _ in range(args.kinds)
        ])
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        # self.upsample = nn.ModuleList([
        #         common.Upsampler(conv, args.scale, n_feats, act=False)
        #                               for _ in range(args.kinds)
        # ])
        self.upsample = common.Upsampler(conv, scale, n_feats, act=False)
        m_tail = [conv(n_feats, args.n_colors, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res = self.pre_process(x)
        res += x
        x = self.upsample(res)
        x = self.tail(x)
        x = self.add_mean(x)
        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

