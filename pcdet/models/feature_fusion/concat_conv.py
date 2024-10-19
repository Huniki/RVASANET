import torch
import torch.nn as nn

import torch
import torch.nn as nn
import math

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Fusion_Layer(nn.Module):
    def __init__(self, in_channels,out_channels,use_attn,use_reference_pts):
        super(Fusion_Layer, self).__init__()

        self.fusion_conv = nn.Conv2d(in_channels,out_channels,1)
        # self.fusion_bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        # self.fusion_relu = nn.ReLU()
        self.use_attn = use_attn
        self.use_reference_pts = use_reference_pts
        # self.fusion_module = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            # nn.ReLU(),
        # )
        if self.use_attn:
            self.attn = se_block(in_channels)
            # pass

    def forward(self, batch_dict):
        seg_features = batch_dict['seg_features']
        spatial_features = batch_dict['spatial_features']

        fusion_features = torch.cat([spatial_features,seg_features],dim=1)
        if self.use_attn:
            fusion_features = self.attn(fusion_features)
        # fusion_features = self.fusion_module(fusion_features)
        fusion_features = self.fusion_conv(fusion_features)
        if self.use_reference_pts:
            empty_features = batch_dict['empty_features']
            batch_dict['spatial_features'] = fusion_features + empty_features
        else:
            batch_dict['spatial_features'] = fusion_features
        return batch_dict



if __name__=='__main__':
    pass
    # inputs = torch.randn((1,128,320,320))
    # channel_att = SELayer(128)
    # outputs = channel_att(inputs)
    # print(outputs.shape)