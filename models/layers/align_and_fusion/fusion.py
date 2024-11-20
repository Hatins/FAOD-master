import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..maxvit.maxvit import MLP, get_act_layer
from .cbam import ChannelGate, SpatialGate

class Conv_BN_ReLU(nn.Module):
    def __init__(self, input_channels, output_channles,
                 kernel_size=3, stride=1, padding=1,
                 bias=True):
        super().__init__()
        self.CBR = nn.Sequential(
            nn.Conv2d(input_channels, output_channles,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(output_channles),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.CBR(x)

class Cat_Fusion(nn.Module):
    def __init__(self, img_feature_channels, ev_feature_channels,
                 output_channels):
        super().__init__()

        assert img_feature_channels == ev_feature_channels
        self.cat_CBR = Conv_BN_ReLU(output_channels * 2, output_channels,
                                    kernel_size=3, stride=1, padding=1)

    def forward(self, frame_coarse_feature, event_coarse_feature):

        output_feature = th.cat([frame_coarse_feature, event_coarse_feature], dim = 1)
        output_feature = self.cat_CBR(output_feature)

        return output_feature

class Cross_mamba(nn.Module):
    def __init__(self, img_feature_channels, ev_feature_channels,
                 output_channels):
        super().__init__()


    
    def forward(self, frame_coarse_feature, event_coarse_feature):
        print()




class Cross_cbam(nn.Module):
    def __init__(self, img_feature_channels, ev_feature_channels,
                 output_channels):
        super().__init__()

        self.frame_c = output_channels
        self.event_c = output_channels

        ## 1. Modality weighting
        self.frame_CBR = Conv_BN_ReLU(img_feature_channels, self.frame_c,
                                      kernel_size=3, stride=1, padding=1)
        self.event_CBR = Conv_BN_ReLU(ev_feature_channels, self.event_c,
                                      kernel_size=3, stride=1, padding=1)
        
        ## 2. Channel attension
        self.frame_CA = ChannelGate(self.frame_c)
        self.event_CA = ChannelGate(self.event_c)

        ## 3. spatial attension
        self.frame_SA = SpatialGate()
        self.event_SA = SpatialGate()

        self.output_conv = Conv_BN_ReLU(output_channels * 2, output_channels,
                                      kernel_size=3, stride=1, padding=1)


    def forward(self, frame_coarse_feature, event_coarse_feature):
        frame_feature = self.frame_CBR(frame_coarse_feature)  # [B, 32, H, W]
        event_feature = self.event_CBR(event_coarse_feature)  # [B, 72, H, W]

        feature_f_e = frame_feature + frame_feature*event_feature
        feature_e_f = event_feature + event_feature*frame_feature

        feature_f_e_CA = self.event_CA(feature_e_f) * feature_f_e + feature_f_e
        feature_e_f_CA = self.frame_CA(feature_f_e) * feature_e_f + feature_e_f

        rgb2 = self.event_SA(feature_e_f_CA) * feature_f_e_CA + feature_f_e_CA
        evt2 = self.frame_SA(feature_f_e_CA) * feature_e_f_CA + feature_e_f_CA

        mul_out = th.mul(rgb2, evt2)

        # max_rgb = th.reshape(rgb2,[rgb2.shape[0],1,rgb2.shape[1],rgb2.shape[2],rgb2.shape[3]])
        # max_evt = th.reshape(evt2,[evt2.shape[0],1,evt2.shape[1],evt2.shape[2],evt2.shape[3]])
        # max_cat = th.cat((max_rgb, max_evt), dim=1)
        # max_out = max_cat.max(dim=1)[0]
        max_out = th.max(rgb2, evt2)

        out = self.output_conv(th.cat((mul_out, max_out), dim=1))
        return out
    
class Seletive_Feature_fusion(nn.Module):
    def __init__(self, img_feature_channels, ev_feature_channels,
                 output_channels):
        super().__init__()

        self.frame_c = output_channels
        self.event_c = output_channels

        self.frame_CBR = Conv_BN_ReLU(img_feature_channels, self.frame_c,
                                      kernel_size=3, stride=1, padding=1)
        self.event_CBR = Conv_BN_ReLU(ev_feature_channels, self.event_c,
                                      kernel_size=3, stride=1, padding=1)
        
        self.weight_frame_CBR = Conv_BN_ReLU(output_channels * 2, output_channels,
                                      kernel_size=3, stride=1, padding=1)
        
        self.weight_event_CBR = Conv_BN_ReLU(output_channels * 2, output_channels,
                                      kernel_size=3, stride=1, padding=1)

        # self.output_MLP = MLP(
        #             dim = output_channels,
        #             channel_last=True,
        #             expansion_ratio = 4,
        #             act_layer = get_act_layer('gelu'),
        #             gated = 0,
        #             bias = True,
        #             drop_prob = 0) 

        self.output_CBR = Conv_BN_ReLU(output_channels, output_channels,
                                      kernel_size=3, stride=1, padding=1)    

    def forward(self, frame_coarse_feature, event_coarse_feature):

        #enhance the similar feature 

        frame_feature = self.frame_CBR(frame_coarse_feature)  
        event_feature = self.event_CBR(event_coarse_feature)  

        frame_feature_weighted = frame_feature + frame_feature*event_feature
        event_feature_weighted = event_feature + event_feature*frame_feature

        cat_feature = th.cat([frame_feature_weighted, event_feature_weighted], dim = 1) 

        #data dependently select feature

        weight_frame = self.weight_frame_CBR(cat_feature)
        weight_event = self.weight_event_CBR(cat_feature)

        output_feature = weight_frame * frame_feature_weighted + weight_event * event_feature_weighted
        output_feature = self.output_CBR(output_feature)

        return output_feature
    


class Cross_wsam(nn.Module):
    def __init__(self, img_feature_channels, ev_feature_channels,
                 output_channels):
        super().__init__()

        self.frame_c = output_channels
        self.event_c = output_channels

        self.se = ChannelGate(img_feature_channels+ev_feature_channels)

        self.frame_weighted_conv = nn.Conv2d(img_feature_channels+ev_feature_channels, self.frame_c, 3, 1, 1)
        self.frame_selected_conv = nn.Sequential(
            nn.Conv2d(img_feature_channels+ev_feature_channels, self.frame_c, 3, 1, 1),
            nn.Sigmoid()
        )

        self.event_weighted_conv = nn.Conv2d(img_feature_channels+ev_feature_channels, self.event_c, 3, 1, 1)
        self.event_selected_conv = nn.Sequential(
            nn.Conv2d(img_feature_channels+ev_feature_channels, self.event_c, 3, 1, 1),
            nn.Sigmoid()
        )
        
        ## 2. Channel attension
        self.frame_CA = ChannelGate(self.frame_c)
        self.event_CA = ChannelGate(self.event_c)

        ## 3. spatial attension
        self.frame_SA = SpatialGate()
        self.event_SA = SpatialGate()

        self.output_conv = Conv_BN_ReLU(output_channels * 2, output_channels,
                                      kernel_size=3, stride=1, padding=1)


    def forward(self, frame_feature, event_feature):
        cat_feature = th.cat([frame_feature, event_feature], dim=1)
        cat_feature = self.se(cat_feature)

        frame_feature = frame_feature + self.frame_weighted_conv(cat_feature)
        frame_feature = frame_feature * self.frame_selected_conv(cat_feature)

        event_feature = event_feature + self.event_weighted_conv(cat_feature)
        event_feature = event_feature * self.event_selected_conv(cat_feature)

        feature_f_e_CA = self.event_CA(event_feature) * frame_feature + frame_feature
        feature_e_f_CA = self.frame_CA(frame_feature) * event_feature + event_feature

        rgb2 = self.event_SA(feature_e_f_CA) * feature_f_e_CA + feature_f_e_CA
        evt2 = self.frame_SA(feature_f_e_CA) * feature_e_f_CA + feature_e_f_CA

        mul_out = th.mul(rgb2, evt2)

        # max_rgb = th.reshape(rgb2,[rgb2.shape[0],1,rgb2.shape[1],rgb2.shape[2],rgb2.shape[3]])
        # max_evt = th.reshape(evt2,[evt2.shape[0],1,evt2.shape[1],evt2.shape[2],evt2.shape[3]])
        # max_cat = th.cat((max_rgb, max_evt), dim=1)
        # max_out = max_cat.max(dim=1)[0]
        max_out = th.max(rgb2, evt2)

        out = self.output_conv(th.cat((mul_out, max_out), dim=1))
        return out
    
