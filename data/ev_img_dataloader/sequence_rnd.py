from pathlib import Path

from data.ev_img_dataloader.labels import SparselyBatchedObjectLabels
from data.ev_img_dataloader.sequence_base import SequenceBase
from data.utils.types import DataType, DatasetType, LoaderDataDictGenX
from utils.timers import TimerDummy as Timer
import ipdb
import numpy as np
import random

class SequenceForRandomAccess(SequenceBase):
    def __init__(self,
                 path: Path,
                 ev_representation_name: str,
                 sequence_length: int,
                 dataset_type: DatasetType,
                 downsample_by_factor_2: bool,
                 only_load_end_labels: bool,
                 time_flip=False,
                 unpair_prob = 0.0,
                 min_max_drift = [0, 0],
                 label_shift=True,
                 image_shift=False):
        super().__init__(path=path,
                         ev_representation_name=ev_representation_name,
                         sequence_length=sequence_length,
                         dataset_type=dataset_type,
                         downsample_by_factor_2=downsample_by_factor_2,
                         only_load_end_labels=only_load_end_labels)

        self.start_idx_offset = None
        self.unpair_prob = unpair_prob
        self.min_max_drift = min_max_drift
        # print('self.objframe_idx_2_repr_idx: ', self.objframe_idx_2_repr_idx)
        self.objframe_idx_2_repr_idx = self.objframe_idx_2_repr_idx[:-1]    # TODO: new

        for objframe_idx, repr_idx in enumerate(self.objframe_idx_2_repr_idx):
            if repr_idx - self.seq_len + 1 >= 0:
                # We can fit the sequence length to the label
                self.start_idx_offset = objframe_idx
                break
        if self.start_idx_offset is None:
            # This leads to actual length of 0:
            self.start_idx_offset = len(self.label_factory)
        # self.length = len(self.label_factory) - self.start_idx_offset
        # assert len(self.label_factory) == len(self.objframe_idx_2_repr_idx)
        # print('self.start_idx_offset: ', self.start_idx_offset) # 10
        self.length = len(self.objframe_idx_2_repr_idx) - self.start_idx_offset
        assert len(self.label_factory) == len(self.objframe_idx_2_repr_idx) + 1

        # Useful for weighted sampler that is based on label statistics:
        self._only_load_labels = False

        self.time_flip = time_flip
        self.max_idx = len(self.objframe_idx_2_repr_idx)

        self.label_shift = label_shift
        self.image_shift = image_shift


    def __len__(self):
        return self.length
    

    def __getitem__(self, index: int):
        return self.getitem_with_guaranteed_labels(index)
    

    def getitem_with_guaranteed_labels(self, index: int) -> LoaderDataDictGenX:
        
        if np.random.rand() < self.unpair_prob: #Always do that due to the random loading
            min_drift = self.min_max_drift[0]
            max_drift = self.min_max_drift[1]
            self.drift = random.randint(min_drift, max_drift)
        else:
            self.drift = 0

        corrected_idx = index + self.start_idx_offset
        labels_repr_idx = self.objframe_idx_2_repr_idx[corrected_idx]

        end_idx = labels_repr_idx + 1
        start_idx = end_idx - self.seq_len
        assert_msg = f'{self.ev_repr_file=}, {self.start_idx_offset=}, {start_idx=}, {end_idx=}'
        assert start_idx >= 0, assert_msg

        labels = list()
        for repr_idx in range(start_idx, end_idx):
            # print('self.only_load_end_labels: ', self.only_load_end_labels)   # False
            if self.only_load_end_labels and repr_idx < end_idx - 1:
                labels.append(None)
            else:
                if not self.time_flip:
                    # labels.append(self._get_labels_from_repr_idx(repr_idx+1))
                    labels.append(self._get_labels_from_repr_idx(repr_idx + int(self.label_shift)))
                else:
                    # labels.append(self._get_labels_from_repr_idx(repr_idx))
                    raise NotImplementedError


        sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=labels) #TODO
        if self._only_load_labels:
            return {DataType.OBJLABELS_SEQ: sparse_labels}

        if sum([element is None for element in sparse_labels]) == len(sparse_labels):
            # print('index: ', index)
            return self.getitem_with_guaranteed_labels(index+1)
        else:
            with Timer(timer_name='read ev reprs'):
                ev_repr = self._get_event_repr_torch(start_idx=start_idx, end_idx=end_idx, time_flip=self.time_flip)
                if start_idx >= self.drift:
                    img = self._get_img_torch(start_idx=start_idx + int(self.image_shift) - self.drift, 
                                              end_idx=end_idx + int(self.image_shift) - self.drift, time_flip=self.time_flip)
                else:
                    img = self._get_img_torch(start_idx=start_idx + int(self.image_shift), 
                                              end_idx=end_idx + int(self.image_shift), time_flip=self.time_flip)
            assert len(sparse_labels) == len(ev_repr) == len(img)

            is_first_sample = True  # Due to random loading
            is_padded_mask = [False] * len(ev_repr)
            out = {
                DataType.EV_REPR: ev_repr,
                DataType.IMAGE: img,    # TODO: new
                DataType.OBJLABELS_SEQ: sparse_labels,
                DataType.IS_FIRST_SAMPLE: is_first_sample,
                DataType.IS_PADDED_MASK: is_padded_mask,
                DataType.DRIFT: self.drift
            }

            # if sum([element is None for element in sparse_labels]) == len(sparse_labels):
            #     # print('index: ', index)
            #     return self.getitem_with_guaranteed_labels(index+1)

            return out
    

    def is_only_loading_labels(self) -> bool:
        return self._only_load_labels

    def only_load_labels(self):
        self._only_load_labels = True

    def load_everything(self):
        self._only_load_labels = False
