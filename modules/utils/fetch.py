import pytorch_lightning as pl
from omegaconf import DictConfig

from data.data_module.ev_img_data_moudle import DataModule as ev_img_data_moudle



def fetch_model_module(config: DictConfig) -> pl.LightningModule:
    model_type = config.model.backbone.type
    if model_type == 'event':
        from modules.detection_event import Module as rnn_det_module
    elif model_type == 'fusion':
        from modules.detection_fusion import Module as rnn_det_module  
    elif model_type == 'frame':
        from modules.detection_frame import Module as rnn_det_module
    else:
        raise NotImplementedError   
     
    return rnn_det_module(config)



def fetch_data_module(config: DictConfig) -> pl.LightningDataModule:
    batch_size_train = config.batch_size.train
    batch_size_eval = config.batch_size.eval
    num_workers_generic = config.hardware.get('num_workers', None)
    num_workers_train = config.hardware.num_workers.get('train', num_workers_generic)
    num_workers_eval = config.hardware.num_workers.get('eval', num_workers_generic)
    dataset_str = config.dataset.name
    if dataset_str in {'pku_fusion','dsec'}:
        return ev_img_data_moudle(config.dataset,
                                num_workers_train=num_workers_train,
                                num_workers_eval=num_workers_eval,
                                batch_size_train=batch_size_train,
                                batch_size_eval=batch_size_eval)
    raise NotImplementedError
