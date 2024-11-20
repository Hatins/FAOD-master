from omegaconf import DictConfig


def  build_recurrent_backbone(backbone_cfg: DictConfig):
    #Don't mind some naming questions here
    name = backbone_cfg.name
    backbone_type = backbone_cfg.backbone_type
    if name == 'forward_fusion':
        if backbone_type == 'resnet':
            from .resnet_rnn_forward_fusion import RNNDetector as MaxViTRNNDetector
        elif backbone_type == 'darknet':
            from .darknet_rnn_forward_fusion import RNNDetector as MaxViTRNNDetector
        elif backbone_type == 'maxvit':
            from .darknet_rnn_forward_fusion import RNNDetector as MaxViTRNNDetector
        elif backbone_type == 'swin':
            from .swin_rnn_forward_fusion import RNNDetector as MaxViTRNNDetector
    elif name == 'overall_fusion':
        from .darknet_rnn_overall_fusion import RNNDetector as MaxViTRNNDetector
    elif name == 'single_modal':
        from .maxvit_rnn_single_modal import RNNDetector as MaxViTRNNDetector
    else:
        raise NotImplementedError
    
    return MaxViTRNNDetector(backbone_cfg)
 