from pathlib import Path
from typing import List, Optional, Union, Tuple

import h5py
try:
    import hdf5plugin
except ImportError:
    pass
import numpy as np
import torch
from omegaconf import DictConfig
from torchdata.datapipes.iter import IterDataPipe

import random
from data.ev_img_dataloader.labels import SparselyBatchedObjectLabels
from data.ev_img_dataloader.sequence_base import SequenceBase, get_objframe_idx_2_repr_idx
from data.utils.augmentor import RandomSpatialAugmentorGenX
from data.utils.types import DataType, DatasetType, LoaderDataDictGenX
from utils.timers import TimerDummy as Timer


def _scalar_as_1d_array(scalar: Union[int, float]):
    return np.atleast_1d(scalar)


def _get_ev_repr_range_indices(indices: np.ndarray, max_len: int) -> List[Tuple[int, int]]:
    """
    Computes a list of index ranges based on the input array of indices and a maximum length.
    The index ranges are computed such that the difference between consecutive indices
    should not exceed the maximum length (max_len).

    Parameters:
    -----------
    indices : np.ndarray
        A NumPy array of indices, where the indices are sorted in ascending order.
    max_len : int
        The maximum allowed length between consecutive indices.

    Returns:
    --------
    out : List[Tuple[int, int]]
        A list of tuples, where each tuple contains two integers representing the start and
        stop indices of the range.
    """
    meta_indices_stop = np.flatnonzero(np.diff(indices) > max_len)

    meta_indices_start = np.concatenate((np.atleast_1d(0), meta_indices_stop + 1))
    meta_indices_stop = np.concatenate((meta_indices_stop, np.atleast_1d(len(indices) - 1)))

    out = list()
    for meta_idx_start, meta_idx_stop in zip(meta_indices_start, meta_indices_stop):
        idx_start = max(indices[meta_idx_start] - max_len + 1, 0)
        idx_stop = indices[meta_idx_stop] + 1
        out.append((idx_start, idx_stop))
    return out


class SequenceForIter(SequenceBase):
    def __init__(self,
                 path: Path,
                 ev_representation_name: str,
                 sequence_length: int,
                 dataset_type: DatasetType,
                 downsample_by_factor_2: bool,
                 range_indices: Optional[Tuple[int, int]] = None,
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
                         only_load_end_labels=False)
        
        self.unpair_prob = unpair_prob
        self.min_max_drift = min_max_drift

        with h5py.File(str(self.ev_repr_file), 'r') as h5f:
            num_ev_repr = h5f['data'].shape[0]
        # with h5py.File(str(self.image_h5_file), 'r') as h5f:
        #     num_image = h5f['data'].shape[0]
        # assert num_ev_repr + 1 == num_image

    
        if range_indices is None:
            repr_idx_start = max(self.objframe_idx_2_repr_idx[0] - sequence_length + 1, 0)
            repr_idx_stop = num_ev_repr
        else:
            repr_idx_start, repr_idx_stop = range_indices
        
        # Set start idx such that the first label is no further than the last timestamp of the first sample sub-sequence
        min_start_repr_idx = max(self.objframe_idx_2_repr_idx[0] - sequence_length + 1, 0)
        assert 0 <= min_start_repr_idx <= repr_idx_start < repr_idx_stop <= num_ev_repr, \
            f'{min_start_repr_idx=}, {repr_idx_start=}, {repr_idx_stop=}, {num_ev_repr=}, {path=}'

        self.idx_start_debug = repr_idx_start

       
        if not time_flip:
            # self.start_indices = list(range(repr_idx_start+self.drift, repr_idx_stop-1, sequence_length))
            # self.stop_indices = self.start_indices[1:] + [repr_idx_stop-1]

            self.start_indices = list(range(repr_idx_start, repr_idx_stop, sequence_length))
            #self.start_indices = list(range(repr_idx_start, repr_idx_stop, sequence_length))
            self.stop_indices = self.start_indices[1:] + [repr_idx_stop]
        else:
            raise NotImplementedError
            # self.start_indices = list(range(repr_idx_stop-self.drift, repr_idx_start+1, -sequence_length))
            # self.stop_indices = self.start_indices[1:] + [repr_idx_start+1]

        self.length = len(self.start_indices)

        self._padding_representation = None

        self.time_flip = time_flip
        self.label_shift = label_shift
        self.image_shift = image_shift
        self.path_for_debug = path    # for test 


    @staticmethod
    def get_sequences_with_guaranteed_labels(
            path: Path,
            ev_representation_name: str,
            sequence_length: int,
            dataset_type: DatasetType,
            downsample_by_factor_2: bool,
            time_flip: bool,
            unpair_prob : float,
            min_max_drift : list,
            label_shift: bool,
            image_shift: bool
    ) -> List['SequenceForIter']:
        """Generate sequences such that we do always have labels within each sample of the sequence
        This is required for training such that we are guaranteed to always have labels in the training step.
        However, for validation we don't require this if we catch the special case.
        """
        objframe_idx_2_repr_idx = get_objframe_idx_2_repr_idx(
            path=path, ev_representation_name=ev_representation_name)
        # max diff for repr idx is sequence length
        range_indices_list = _get_ev_repr_range_indices(indices=objframe_idx_2_repr_idx[:-1], max_len=sequence_length)


        sequence_list = list()
        for range_indices in range_indices_list:
            sequence_list.append(
                SequenceForIter(path=path,
                                ev_representation_name=ev_representation_name,
                                sequence_length=sequence_length,
                                dataset_type=dataset_type,
                                downsample_by_factor_2=downsample_by_factor_2,
                                range_indices=range_indices,
                                time_flip=time_flip,
                                unpair_prob = unpair_prob,
                                min_max_drift = min_max_drift,
                                label_shift=label_shift,
                                image_shift=image_shift)
            )
        return sequence_list

    @property
    def ev_padding_representation(self) -> torch.Tensor:
        if self._padding_representation is None:
            ev_repr = self._get_event_repr_torch(start_idx=0, end_idx=1)[0]
            self._ev_padding_representation = torch.zeros_like(ev_repr)
        return self._ev_padding_representation
    
    @property
    def img_padding_representation(self) -> torch.Tensor:
        if self._padding_representation is None:
            img = self._get_img_torch(start_idx=0, end_idx=1)[0]
            self._img_padding_representation = torch.zeros_like(img)
        return self._img_padding_representation

    def get_fully_padded_sample(self) -> LoaderDataDictGenX:
        is_first_sample = False
        is_padded_mask = [True] * self.seq_len
        drift_mask = 0
        ev_repr = [self.ev_padding_representation] * self.seq_len
        img = [self.img_padding_representation] * self.seq_len
        labels = [None] * self.seq_len
        sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=labels)
        out = {
            DataType.EV_REPR: ev_repr,
            DataType.IMAGE: img,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: is_first_sample,
            DataType.IS_PADDED_MASK: is_padded_mask,
            DataType.DRIFT: drift_mask,
        }
        return out

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
    
        is_first_sample = True if index == 0 else False

        if is_first_sample:  #modified the value of the draft at the first sample
            if np.random.rand() < self.unpair_prob:
                min_drift = self.min_max_drift[0]
                max_drift = self.min_max_drift[1]
                self.drift = random.randint(min_drift, max_drift)
            else:
                self.drift = 0

        start_idx = self.start_indices[index]
        end_idx = self.stop_indices[index]

        # sequence info ###
        if not self.time_flip:
            sample_len = end_idx - start_idx
            assert self.seq_len >= sample_len > 0, f'{self.seq_len=}, {sample_len=}, {start_idx=}, {end_idx=}, ' \
                                                   f'\n{self.start_indices=}\n{self.stop_indices=}'
        else:
            sample_len = start_idx - end_idx
            assert self.seq_len >= sample_len > 0, f'{self.seq_len=}, {sample_len=}, {start_idx=}, {end_idx=}, ' \
                                                   f'\n{self.start_indices=}\n{self.stop_indices=}'

        is_padded_mask = [False] * sample_len
        ###################

        # event representations ###
        with Timer(timer_name='read ev reprs'):
            if not self.time_flip:
                ev_repr = self._get_event_repr_torch(start_idx=start_idx, end_idx=end_idx)
                if start_idx - self.drift >= 0:
                    img = self._get_img_torch(start_idx=start_idx+int(self.image_shift)-self.drift, end_idx=end_idx+int(self.image_shift)-self.drift) # TODO: new
                else:
                    img = self._get_img_torch(start_idx=start_idx+int(self.image_shift), end_idx=end_idx+int(self.image_shift)) # TODO: new
            else:
                # ev_repr = self._get_event_repr_torch(start_idx=end_idx-1, end_idx=start_idx-1, time_flip=self.time_flip)
                # img = self._get_img_torch(start_idx=end_idx+self.drift, end_idx=start_idx+self.drift, time_flip=self.time_flip)  # TODO: new
                raise NotImplementedError
        assert len(ev_repr) == sample_len == len(img)
        ###########################

        # labels ###
        labels = list()
        label_shift = int(self.label_shift)
        if not self.time_flip:
            # for repr_idx in range(start_idx+1, end_idx+1):
            for repr_idx in range(start_idx + label_shift, end_idx + label_shift):
                labels.append(self._get_labels_from_repr_idx(repr_idx))
        else:
            # for repr_idx in range(end_idx-1, start_idx-1):
            for repr_idx in range(end_idx - label_shift, start_idx - label_shift):
                labels.append(self._get_labels_from_repr_idx(repr_idx))
            labels = labels[::-1]
        assert len(labels) == len(ev_repr)

        ############

        # apply padding (if necessary) ###
        if sample_len < self.seq_len:
            padding_len = self.seq_len - sample_len

            is_padded_mask.extend([True] * padding_len)
            ev_repr.extend([self.ev_padding_representation] * padding_len)
            img.extend([self.img_padding_representation] * padding_len)
            labels.extend([None] * padding_len)
        ##################################

        # convert labels to sparse labels for datapipes and dataloader
        sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=labels)

        out = {
            DataType.EV_REPR: ev_repr,
            DataType.IMAGE: img,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: is_first_sample,
            DataType.IS_PADDED_MASK: is_padded_mask,
            DataType.DRIFT: self.drift
        }
        return out


class RandAugmentIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, dataset_config: DictConfig):
        super().__init__()
        self.source_dp = source_dp

        resolution_hw = tuple(dataset_config.resolution_hw)
        assert len(resolution_hw) == 2
        ds_by_factor_2 = dataset_config.downsample_by_factor_2
        if ds_by_factor_2:
            resolution_hw = tuple(x // 2 for x in resolution_hw)

        augm_config = dataset_config.data_augmentation
        self.spatial_augmentor = RandomSpatialAugmentorGenX(
            dataset_hw=resolution_hw,
            automatic_randomization=False,
            augm_config=augm_config.stream)

    def __iter__(self):
        self.spatial_augmentor.randomize_augmentation()
        for x in self.source_dp:
            yield self.spatial_augmentor(x)