import torch
from .EgF import EgF, ALIGN
import torch.nn as nn
import os
import torch.nn.functional as F
import sys

from .adain import adaptive_instance_normalization as adain


class Feature_wrapper(nn.Module):
    def __init__(self, img_feature_channels, ev_feature_channels):
        super().__init__()

        self.guide = EgF(inplanes=ev_feature_channels, planes=ev_feature_channels//8)

        self.align = ALIGN(input_dim=ev_feature_channels)

        self.conv = nn.Conv2d(img_feature_channels+ev_feature_channels, img_feature_channels, 1)

    
    def forward(self, frame_feature, event_feature, offset=None):
        frame_feature, fea_e1 = self.guide(frame_feature, event_feature)
        fea_e1 = fea_e1 + event_feature

        transfer_frame_feature = adain(frame_feature, fea_e1)
        cat_fea = self.conv(torch.cat([transfer_frame_feature, fea_e1], dim=1))

        align_feature = self.align(cat_fea, frame_feature, offset)

        return align_feature
    

class Blur_aug(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.conv = nn.Conv2d(input_channels*2, output_channels, kernel_size=1)

    
    def forward(self, feature, ratio=2):
        H, W = feature.shape[2:]
        feature_d = F.max_pool2d(feature, kernel_size=(ratio, ratio))
        feature_d_u = F.interpolate(feature_d, size=(H, W), mode='bilinear')
        return self.conv(torch.cat([feature, feature_d_u], dim=1))




    

