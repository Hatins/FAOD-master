from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from data.utils.types import FeatureMap, BackboneFeatures, LstmState, LstmStates
from models.layers.rnn import DWSConvLSTM2d
from models.layers.maxvit.maxvit import (
    PartitionAttentionCl,
    nhwC_2_nChw,
    nChw_2_nhwC,
    get_downsample_layer_Cf2Cl,
    PartitionType)

from models.layers.darknet.modules import (
    Temporal_Active_Focus_connect,
    BaseConv,
    ResLayer,
    SPPBottleneck,
    Focus_yolo,
    CSPLayer,

    )

from models.layers.align_and_fusion.align import Feature_wrapper, Blur_aug
from models.layers.align_and_fusion.cross_mamba import CROSS_Mamba_Fusion
from models.layers.align_and_fusion.fusion import Cross_cbam, Seletive_Feature_fusion, Cat_Fusion, Cross_wsam

from .base import BaseDetector
import ipdb

class RNNDetector(BaseDetector):
    def __init__(self, mdl_config: DictConfig):
        super().__init__()

        ev_in_channels = mdl_config.ev_input_channels
        img_in_channels = mdl_config.img_input_channels

        embed_dim = mdl_config.embed_dim
        dim_multiplier_per_stage = tuple(mdl_config.dim_multiplier)
        num_blocks_per_stage = tuple(mdl_config.num_blocks)
        T_max_chrono_init_per_stage = tuple(mdl_config.T_max_chrono_init)
        enable_masking = mdl_config.enable_masking


        num_stages = len(num_blocks_per_stage)
        assert num_stages == 4

        assert isinstance(embed_dim, int)
        assert num_stages == len(dim_multiplier_per_stage)
        assert num_stages == len(num_blocks_per_stage)
        assert num_stages == len(T_max_chrono_init_per_stage)

        ###### Compile if requested ######
        compile_cfg = mdl_config.get('compile', None)
        if compile_cfg is not None:
            compile_mdl = compile_cfg.enable
            if compile_mdl and th_compile is not None:
                compile_args = OmegaConf.to_container(compile_cfg.args, resolve=True, throw_on_missing=True)
                self.forward = th_compile(self.forward, **compile_args)
            elif compile_mdl:
                print('Could not compile backbone because torch.compile is not available')
        ##################################
                
        patch_size = mdl_config.stem.patch_size
        stride = 1
        self.stage_dims = [embed_dim * x for x in dim_multiplier_per_stage]

        self.stages = nn.ModuleList()
        self.strides = []

        for stage_idx, (num_blocks, T_max_chrono_init_stage) in \
            enumerate(zip(num_blocks_per_stage, T_max_chrono_init_per_stage)):
            spatial_downsample_factor = patch_size if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]
            enable_masking_in_stage = enable_masking and stage_idx == 0
            stage = RNNFusionDetectorStage(
                    stage_idx = stage_idx,
                    dim_in_ev=ev_in_channels,
                    dim_in_img = img_in_channels,
                    stage_dim=stage_dim,
                    spatial_downsample_factor=spatial_downsample_factor,
                    num_blocks=num_blocks,
                    enable_token_masking=enable_masking_in_stage,
                    T_max_chrono_init=T_max_chrono_init_stage,
                    stage_cfg=mdl_config.stage)
            stride = stride * spatial_downsample_factor
            self.strides.append(stride)

            ev_in_channels = stage_dim
            img_in_channels = stage_dim
            self.stages.append(stage)
    
        self.num_stages = num_stages


    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.stage_dims[stage_idx] for stage_idx in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.strides[stage_idx] for stage_idx in stage_indices)
        
    def forward(self, ev_input: th.Tensor, img_input: th.tensor, prev_states: Optional[LstmStates] = None, token_mask: Optional[th.Tensor] = None) \
            -> Tuple[BackboneFeatures, LstmStates]:
        if prev_states is None:
            prev_states = [None] * self.num_stages
        assert len(prev_states) == self.num_stages
        states: LstmStates = list()
        output: Dict[int, FeatureMap] = {}
        for stage_idx, stage in enumerate(self.stages):
            if stage_idx == 0:
                x, ev_input, img_input, state = stage(ev_input, img_input, None, prev_states[stage_idx], token_mask if stage_idx == 0 else None)
            else:
                x, ev_input, img_input, state = stage(ev_input, img_input, x, prev_states[stage_idx], token_mask if stage_idx == 0 else None)
            
            states.append(state)
            stage_number = stage_idx + 1
            output[stage_number] = x
            loss_align = 0
        return output, states, loss_align, 0, 0

class MaxVitAttentionPairCl(nn.Module):
    def __init__(self,
                 dim: int,
                 skip_first_norm: bool,
                 attention_cfg: DictConfig):
        super().__init__()

        self.att_window = PartitionAttentionCl(dim=dim,
                                               partition_type=PartitionType.WINDOW,
                                               attention_cfg=attention_cfg,
                                               skip_first_norm=skip_first_norm)
        self.att_grid = PartitionAttentionCl(dim=dim,
                                             partition_type=PartitionType.GRID,
                                             attention_cfg=attention_cfg,
                                             skip_first_norm=False)

    def forward(self, x):
        x = self.att_window(x)
        x = self.att_grid(x)
        return x



     

class RNNFusionDetectorStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output.
    """

    def __init__(self,
                 stage_idx: int,
                 dim_in_ev: int,
                 dim_in_img: int,
                 stage_dim: int,
                 spatial_downsample_factor: int,
                 num_blocks: int,
                 enable_token_masking: bool,
                 T_max_chrono_init: Optional[int],
                 stage_cfg: DictConfig):
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0
        downsample_cfg = stage_cfg.downsample
        lstm_cfg = stage_cfg.lstm
        attention_cfg = stage_cfg.attention
        
        self.base_depth = 3
        
                
        if  stage_idx == 0:
        
            self.ev_layers = nn.Sequential(
                    Focus_yolo(dim_in_ev, stage_dim, ksize=3, act='silu'),
                    self.make_group_layer(stage_dim, stage_dim)
                )

            self.img_layers = nn.Sequential(
                    Focus_yolo(dim_in_img, stage_dim, ksize=3, act='silu'),
                    self.make_group_layer(stage_dim, stage_dim)
            )
        
        elif stage_idx == 1 or stage_idx == 2:
            self.ev_layers = self.make_group_layer(dim_in_ev, stage_dim)
            self.img_layers = self.make_group_layer(dim_in_img, stage_dim)
        
        elif stage_idx == 3:
            self.ev_layers = self.make_spp_block(dim_in_ev, stage_dim)
            self.img_layers = self.make_spp_block(dim_in_img, stage_dim)


        self.align_block = Feature_wrapper(img_feature_channels = stage_dim,ev_feature_channels = stage_dim)

        self.fusion_block = Cross_cbam(img_feature_channels = stage_dim,ev_feature_channels = stage_dim,
                                           output_channels = stage_dim)

        self.lstm = DWSConvLSTM2d(dim=stage_dim,
                                  dws_conv=lstm_cfg.dws_conv,
                                  dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                  dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                  cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_cat = nn.Conv2d(int(stage_dim * 1.5), stage_dim,
                                      kernel_size=3, stride=1, padding=1)
        
        # self.Conv_BN_ReLU(int(stage_dim * 1.5), stage_dim,
        #                               kernel_size=3, stride=1, padding=1)

        ###### Mask Token ################
        self.mask_token = nn.Parameter(th.zeros(1, 1, 1, stage_dim),
                                       requires_grad=True) if enable_token_masking else None
        if self.mask_token is not None:
            th.nn.init.normal_(self.mask_token, std=.02)
        ##################################

    def make_group_layer(self, input_channel, output_channel, act = "silu"):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return nn.Sequential(
            BaseConv(input_channel, output_channel, 3, 2, act=act),
            CSPLayer(
                output_channel,
                output_channel,
                n=self.base_depth * 3,
                depthwise= False,
                act=act,
            ),
        )


    def make_spp_block(self, input_channel, output_channel, act = "silu"):
        return nn.Sequential(
            BaseConv(input_channel, output_channel, 3, 2, act=act),
            SPPBottleneck(output_channel, output_channel, activation=act),
            CSPLayer(
                output_channel,
                output_channel,
                n=self.base_depth * 3,
                depthwise= False,
                act=act,
            ),
        )
        
    def forward(self, ev_input: th.Tensor,
                img_input: th.Tensor,
                pre_fusion_input: th.Tensor,
                h_and_c_previous: Optional[LstmState] = None,
                token_mask: Optional[th.Tensor] = None) \
            -> Tuple[FeatureMap, LstmState]:
        
   
        ev_input = self.ev_layers(ev_input)
        img_input = self.img_layers(img_input)

        img_input = self.align_block(img_input, ev_input)
        x = self.fusion_block(img_input, ev_input)

        h_c_tuple = self.lstm(x, h_and_c_previous)
        x = h_c_tuple[0]
        if pre_fusion_input != None:
            pre_fusion_input = self.max_pool(pre_fusion_input)
            x = th.cat([x, pre_fusion_input], dim = 1)
            x = self.conv_cat(x)
        align_loss = 0
        return x, ev_input, img_input, h_c_tuple









                
        



    
