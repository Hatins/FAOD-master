from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None
from einops import rearrange
from models.layers.s5.s5_model import S5Block

from data.utils.types import FeatureMap, BackboneFeatures, LstmState, LstmStates
from models.layers.rnn import DWSConvLSTM2d
from models.layers.maxvit.maxvit import (
    PartitionAttentionCl,
    nhwC_2_nChw,
    get_downsample_layer_Cf2Cl,
    PartitionType)

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
        memory_type = mdl_config.memory_type
        self.memory_type = memory_type
        time_scale = mdl_config.time_scale

        embed_dim = mdl_config.embed_dim
        dim_multiplier_per_stage = tuple(mdl_config.dim_multiplier)
        num_blocks_per_stage = tuple(mdl_config.num_blocks)
        T_max_chrono_init_per_stage = tuple(mdl_config.T_max_chrono_init)
        enable_masking = mdl_config.enable_masking

        enable_align = mdl_config.enable_align
        enable_blur_aug = mdl_config.enable_blur_aug
        fusion_type = mdl_config.fusion_type

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

        # first stage
        spatial_downsample_factor_first_stage = patch_size
        stage_dim = self.stage_dims[0]
        enable_masking_in_stage = enable_masking

        initial_stage = RNNFusionDetectorStage(dim_in_ev=ev_in_channels,
                                               dim_in_img=img_in_channels,
                                               stage_dim=stage_dim,
                                               spatial_downsample_factor=spatial_downsample_factor_first_stage,
                                               num_blocks=num_blocks_per_stage[0],
                                               enable_token_masking=enable_masking_in_stage,
                                               T_max_chrono_init=T_max_chrono_init_per_stage[0],
                                               stage_cfg=mdl_config.stage,
                                               enable_align=enable_align,
                                               enable_blur_aug=enable_blur_aug,
                                               fusion_type=fusion_type,
                                               memory_type=memory_type,
                                               time_scale=time_scale,
                                               using_aligned_loss=mdl_config.using_align_loss)

        stride = stride * spatial_downsample_factor_first_stage
        self.strides.append(stride)
        self.stages.append(initial_stage)
        input_dim = stage_dim
        for stage_idx, (num_blocks, T_max_chrono_init_stage) in \
                enumerate(zip(num_blocks_per_stage, T_max_chrono_init_per_stage)):
            if stage_idx == 0:
                continue
            spatial_downsample_factor = patch_size if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]
            enable_masking_in_stage = enable_masking and stage_idx == 0
            stage = RNNDetectorStage(dim_in=input_dim,
                                     stage_dim=stage_dim,
                                     spatial_downsample_factor=spatial_downsample_factor,
                                     num_blocks=num_blocks,
                                     enable_token_masking=enable_masking_in_stage,
                                     T_max_chrono_init=T_max_chrono_init_stage,
                                     stage_cfg=mdl_config.stage,
                                     memory_type=memory_type,
                                     time_scale=time_scale)
            stride = stride * spatial_downsample_factor
            self.strides.append(stride)

            input_dim = stage_dim
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

    def forward(self, ev_input: th.Tensor, img_input: th.tensor, prev_states: Optional[LstmStates] = None,
                token_mask: Optional[th.Tensor] = None,
                training_step=True, drift: Optional[th.Tensor] = None) \
            -> Tuple[BackboneFeatures, LstmStates]:
        if prev_states is None:
            prev_states = [None] * self.num_stages
        assert len(prev_states) == self.num_stages
        states: LstmStates = list()
        output: Dict[int, FeatureMap] = {}
        for stage_idx, stage in enumerate(self.stages):
            if stage_idx == 0:
                x, state, loss_align, feature_alighed, feature_unalighed = stage(ev_input, img_input, prev_states[stage_idx],
                                             token_mask if stage_idx == 0 else None, drift=drift)
            else:
                x, state = stage(x, prev_states[stage_idx], token_mask if stage_idx == 0 else None)
            states.append(state)
            stage_number = stage_idx + 1
            output[stage_number] = x
        return output, states, loss_align, feature_alighed, feature_unalighed


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
                 dim_in_ev: int,
                 dim_in_img: int,
                 stage_dim: int,
                 spatial_downsample_factor: int,
                 num_blocks: int,
                 enable_token_masking: bool,
                 T_max_chrono_init: Optional[int],
                 stage_cfg: DictConfig,
                 enable_align: bool,
                 enable_blur_aug: bool,
                 fusion_type: str,
                 memory_type: str,
                 time_scale: float = 1.0,
                 using_aligned_loss: bool = True):
        self.memory_type = memory_type
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0
        downsample_cfg = stage_cfg.downsample
        lstm_cfg = stage_cfg.lstm
        attention_cfg = stage_cfg.attention
        self.ev_downsample_cf2cl = get_downsample_layer_Cf2Cl(dim_in=dim_in_ev,
                                                              dim_out=stage_dim,
                                                              downsample_factor=spatial_downsample_factor,
                                                              downsample_cfg=downsample_cfg)

        self.img_downsample_cf2cl = get_downsample_layer_Cf2Cl(dim_in=dim_in_img,
                                                               dim_out=stage_dim,
                                                               downsample_factor=spatial_downsample_factor,
                                                               downsample_cfg=downsample_cfg)

        ev_blocks = [MaxVitAttentionPairCl(dim=stage_dim,
                                           skip_first_norm=i == 0 and self.ev_downsample_cf2cl.output_is_normed(),
                                           attention_cfg=attention_cfg) for i in range(num_blocks)]

        img_blocks = [MaxVitAttentionPairCl(dim=stage_dim,
                                            skip_first_norm=i == 0 and self.img_downsample_cf2cl.output_is_normed(),
                                            attention_cfg=attention_cfg) for i in range(num_blocks)]

        self.ev_att_blocks = nn.ModuleList(ev_blocks)

        self.img_att_blocks = nn.ModuleList(img_blocks)

        self.enable_align = enable_align
        self.enable_blur_aug = enable_blur_aug

        self.using_aligned_loss = using_aligned_loss

        if self.enable_align:
            self.align_block = Feature_wrapper(img_feature_channels=stage_dim, ev_feature_channels=stage_dim)

        if self.enable_blur_aug:
            self.blur_aug = Blur_aug(stage_dim, stage_dim)

        if fusion_type == 'selective':
            self.fusion_block = Seletive_Feature_fusion(img_feature_channels=stage_dim, ev_feature_channels=stage_dim,
                                                        output_channels=stage_dim)
        elif fusion_type == 'cross_cbam':
            self.fusion_block = Cross_cbam(img_feature_channels=stage_dim, ev_feature_channels=stage_dim,
                                           output_channels=stage_dim)
        elif fusion_type == 'cat':
            self.fusion_block = Cat_Fusion(img_feature_channels=stage_dim, ev_feature_channels=stage_dim,
                                           output_channels=stage_dim)
        elif fusion_type == 'cross_mamba':
            self.fusion_block = CROSS_Mamba_Fusion(stage_dim, d_state=16, expand=2, dropout=0.0)
        elif fusion_type == 'cross_wsam':
            self.fusion_block = Cross_wsam(img_feature_channels=stage_dim, ev_feature_channels=stage_dim,
                                           output_channels=stage_dim)

        if memory_type == 'lstm':
            self.lstm = DWSConvLSTM2d(dim=stage_dim,
                                      dws_conv=lstm_cfg.dws_conv,
                                      dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                      dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                      cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))

        elif memory_type == 's5':
            self.s5_block = S5Block(
                dim=stage_dim, state_dim=stage_dim, bidir=False, bandlimit=0.5, step_scale=time_scale
            )

        ###### Mask Token ################
        self.mask_token = nn.Parameter(th.zeros(1, 1, 1, stage_dim),
                                       requires_grad=True) if enable_token_masking else None
        if self.mask_token is not None:
            th.nn.init.normal_(self.mask_token, std=.02)
        ##################################

    def forward_ssm(self, ev_input: th.Tensor,
                    img_input: th.Tensor,
                    states: Optional[LstmState] = None,
                    token_mask: Optional[th.Tensor] = None,
                    train_step: bool = True,
                    drift: Optional[th.Tensor] = None) \
            -> Tuple[FeatureMap, LstmState]:
        sequence_length = ev_input.shape[0]
        batch_size = ev_input.shape[1]

        ev_input = rearrange(ev_input, "L B C H W -> (L B) C H W")  # where B' = (L B) is the new batch size
        img_input = rearrange(img_input, "L B C H W -> (L B) C H W")  # where B' = (L B) is the new batch size

        ev_input = self.ev_downsample_cf2cl(ev_input)  # (L B) C H W -> (L B) H W C
        img_input = self.img_downsample_cf2cl(img_input)  # (L B) C H W -> (L B) H W C

        if token_mask is not None:
            assert self.mask_token is not None, "No mask token present in this stage"
            ev_input[token_mask] = self.mask_token
            img_input[token_mask] = self.mask_token

        for ev_blk in self.ev_att_blocks:
            ev_input = ev_blk(ev_input)

        for img_blk in self.img_att_blocks:
            img_input = img_blk(img_input)

        img_input = nhwC_2_nChw(img_input)  # (L B) H W C -> (L B) C H W
        ev_input = nhwC_2_nChw(ev_input)  # (L B) H W C -> (L B) C H W

        if self.enable_align:
            img_input_aligned = self.align_block(img_input, ev_input)

            if self.using_aligned_loss and drift is not None:
                img_input_unaligned = self.align_block(img_input, ev_input, offset=0)
                ## cal loss
                img_input_aligned = rearrange(img_input_aligned, "(L B) C H W -> L B C H W", L=sequence_length)
                img_input_unaligned = rearrange(img_input_unaligned, "(L B) C H W -> L B C H W", L=sequence_length)
                align_loss = 0
                for bb in range(img_input_aligned.shape[1]):
                    img_input_aligned_bb = img_input_aligned[:(sequence_length - drift[bb]), bb]  # [L1, C, H, W]
                    img_input_unaligned_bb = img_input_unaligned[-(sequence_length - drift[bb]):, bb]  # [L1, C, H, W]
                    loss_bb = F.mse_loss(img_input_aligned_bb, img_input_unaligned_bb)
                    align_loss += loss_bb
                align_loss = align_loss / img_input_aligned.shape[1]

                img_input_aligned = rearrange(img_input_aligned, "L B C H W -> (L B) C H W", L=sequence_length)
            else:
                align_loss = 0

        if self.enable_blur_aug:
            img_input_aligned = self.blur_aug(img_input_aligned, ratio=2)

        x = self.fusion_block(img_input_aligned,
                              ev_input)  # [B, C, H, W] + [B, C, H, W] -> [B, C, H, W] #the former is more important

        new_h, new_w = x.shape[2], x.shape[3]

        x = rearrange(x, "(L B) C H W -> (B H W) L C", L=sequence_length)

        if states is None:
            states = self.s5_block.s5.initial_state(batch_size=batch_size * new_h * new_w).to(x.device)
        else:
            states = rearrange(states, "B C H W -> (B H W) C")

        x, states = self.s5_block(x, states)
        x = rearrange(x, "(B H W) L C -> L B C H W", B=batch_size, H=int(new_h), W=int(new_w))

        states = rearrange(states, "(B H W) C -> B C H W", H=new_h, W=new_w)

        return x, states, align_loss

    def forward_rnn(self, ev_input: th.Tensor,
                    img_input: th.Tensor,
                    h_and_c_previous: Optional[LstmState] = None,
                    token_mask: Optional[th.Tensor] = None) \
            -> Tuple[FeatureMap, LstmState]:
        ev_input = self.ev_downsample_cf2cl(ev_input)  # N C H W -> N H W C
        img_input = self.img_downsample_cf2cl(img_input)  # N C H W -> N H W C

        if token_mask is not None:
            assert self.mask_token is not None, 'No mask token present in this stage'
            ev_input[token_mask] = self.mask_token
            img_input[token_mask] = self.mask_token

        for ev_blk in self.ev_att_blocks:
            ev_input = ev_blk(ev_input)

        for img_blk in self.img_att_blocks:
            img_input = img_blk(img_input)

        img_input = nhwC_2_nChw(img_input)
        ev_input = nhwC_2_nChw(ev_input)
        if self.enable_align:
            img_input_aligned = self.align_block(img_input, ev_input)
            
            if self.using_aligned_loss:
                img_unaligned = self.align_block(img_input, ev_input, offset=0)
            else:
                img_unaligned = None

        if self.enable_blur_aug:
 
            raise NotImplementedError

        x = self.fusion_block(img_input_aligned, ev_input)  # [B, C, H, W] + [B, C, H, W] -> [B, C, H, W]
        h_c_tuple = self.lstm(x, h_and_c_previous)
        x = h_c_tuple[0]
        align_loss = 0 #overlook it
        return x, h_c_tuple, align_loss, img_input_aligned, img_unaligned

    def forward(self,
                ev_input: th.Tensor,
                img_input: th.Tensor,
                h_and_c_previous: Optional[LstmState] = None,
                token_mask: Optional[th.Tensor] = None,
                train_step: bool = True,
                drift: Optional[th.Tensor] = None,
                using_aligned_loss: bool = True
                ) -> Tuple[FeatureMap, LstmState]:
        if self.memory_type in ['lstm', 'gru', 'rnn']:
            return self.forward_rnn(ev_input, img_input, h_and_c_previous, token_mask)

        elif self.memory_type in ['s4', 's5', 's6']:
            return self.forward_ssm(ev_input, img_input, h_and_c_previous, token_mask, train_step,
                                    drift=drift)


class RNNDetectorStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output.
    """

    def __init__(self,
                 dim_in: int,
                 stage_dim: int,
                 spatial_downsample_factor: int,
                 num_blocks: int,
                 enable_token_masking: bool,
                 T_max_chrono_init: Optional[int],
                 stage_cfg: DictConfig,
                 memory_type: str = 'lstm',
                 time_scale: float = 1.0,
                 size: list = []):
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0
        self.memory_type = memory_type
        downsample_cfg = stage_cfg.downsample
        lstm_cfg = stage_cfg.lstm
        attention_cfg = stage_cfg.attention

        self.downsample_cf2cl = get_downsample_layer_Cf2Cl(dim_in=dim_in,
                                                           dim_out=stage_dim,
                                                           downsample_factor=spatial_downsample_factor,
                                                           downsample_cfg=downsample_cfg)
        blocks = [MaxVitAttentionPairCl(dim=stage_dim,
                                        skip_first_norm=i == 0 and self.downsample_cf2cl.output_is_normed(),
                                        attention_cfg=attention_cfg) for i in range(num_blocks)]
        self.att_blocks = nn.ModuleList(blocks)

        if memory_type == 'lstm':
            self.lstm = DWSConvLSTM2d(dim=stage_dim,
                                      dws_conv=lstm_cfg.dws_conv,
                                      dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                      dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                      cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))

        elif memory_type == 's5':
            self.s5_block = S5Block(
                dim=stage_dim, state_dim=stage_dim, bidir=False, bandlimit=0.5, step_scale=time_scale
            )

        ###### Mask Token ################
        self.mask_token = nn.Parameter(th.zeros(1, 1, 1, stage_dim),
                                       requires_grad=True) if enable_token_masking else None
        if self.mask_token is not None:
            th.nn.init.normal_(self.mask_token, std=.02)
        ##################################

    def forward_ssm(self,
                    x: th.Tensor,
                    states: Optional[LstmState] = None,
                    token_mask: Optional[th.Tensor] = None,
                    train_step: bool = True,
                    ) -> Tuple[FeatureMap, LstmState]:
        sequence_length = x.shape[0]
        batch_size = x.shape[1]
        x = rearrange(x, "L B C H W -> (L B) C H W")  # where B' = (L B) is the new batch size
        x = self.downsample_cf2cl(x)  # B' C H W -> B' H W C

        if token_mask is not None:
            assert self.mask_token is not None, "No mask token present in this stage"
            x[token_mask] = self.mask_token

        for blk in self.att_blocks:
            x = blk(x)

        x = nhwC_2_nChw(x)  # B' H W C -> B' C H W
        new_h, new_w = x.shape[2], x.shape[3]
        x = rearrange(x, "(L B) C H W -> (B H W) L C", L=sequence_length)

        if states is None:
            states = self.s5_block.s5.initial_state(batch_size=batch_size * new_h * new_w).to(x.device)
        else:
            states = rearrange(states, "B C H W -> (B H W) C")

        x, states = self.s5_block(x, states)
        x = rearrange(x, "(B H W) L C -> L B C H W", B=batch_size, H=int(new_h), W=int(new_w))
        states = rearrange(states, "(B H W) C -> B C H W", H=new_h, W=new_w)

        return x, states

    def forward_rnn(self, x: th.Tensor,
                    h_and_c_previous: Optional[LstmState] = None,
                    token_mask: Optional[th.Tensor] = None) \
            -> Tuple[FeatureMap, LstmState]:
        x = self.downsample_cf2cl(x)  # N C H W -> N H W C
        if token_mask is not None:
            assert self.mask_token is not None, 'No mask token present in this stage'
            x[token_mask] = self.mask_token
        for blk in self.att_blocks:
            x = blk(x)
        x = nhwC_2_nChw(x)  # N H W C -> N C H W
        h_c_tuple = self.lstm(x, h_and_c_previous)
        x = h_c_tuple[0]
        return x, h_c_tuple

    def forward(self, x: th.Tensor,
                h_and_c_previous: Optional[LstmState] = None,
                token_mask: Optional[th.Tensor] = None,
                train_step: bool = True
                ) -> Tuple[FeatureMap, LstmState]:
        if self.memory_type in ['lstm', 'gru', 'rnn']:
            return self.forward_rnn(x, h_and_c_previous, token_mask)

        elif self.memory_type in ['s4', 's5', 's6']:
            return self.forward_ssm(x, h_and_c_previous, token_mask, train_step)















