'''
This script is used to transfer the aedat4 format in SODFormer to the h5 format, which is often used in recent works.
restruction of the outcomes:
-events
    -x
    -y
    -height
    -width
    -p
    -t
-aps
    -image
    -timestamp
    -height
    -width
Author: Yuyang
Time: 2024/03/11
'''
import dv_processing as dv
import cv2 as cv
import h5py
import hdf5plugin
import numpy as np
import torch
import os
import glob
import argparse
from tqdm import tqdm

from davis_utils.buffer import FrameBuffer, EventBuffer

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


pku_scenes = ['normal', 'low_light', 'motion_blur']
pku_splits = ['train', 'val', 'test']


def parse_argument():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## dir params
    parser.add_argument('--scene', type=str,
                        default='all',
                        help='',choices=['normal', 'low_light', 'motion_blur','all'])
    parser.add_argument('--base_file_dir', type=str,
                        default='/data2/zht/fusion_detection/PKU-DATA/raw/',
                        help='The name of a aedat4 dir')
    parser.add_argument('--base_save_dir', type=str,
                        default='/data2/zht/fusion_detection/PKU-H5/',
                        help='')

    parser.add_argument('--show', default=True, help='if show detection and tracking process')
    return parser


def get_reader(file_path):
    assert os.path.exists(file_path), 'The file \'{}\' is not exist'.format(file_path)
    camera_reader = dv.io.MonoCameraRecording(file_path)

    return camera_reader

if __name__ == '__main__':
    ## Get params
    args, _ = parse_argument().parse_known_args(None)
    print(args)


    file_dir_list = []
    save_dir_list = []
    if args.scene != 'all':
        assert args.scene in pku_scenes
        for split in pku_splits:
            file_dir_list.append(os.path.join(args.base_file_dir, split, args.scene))
            save_dir_list.append(os.path.join(args.base_save_dir, split, args.scene))
    elif args.scene == 'all':
        for scene in pku_scenes:
            for split in pku_splits:
                file_dir_list.append(os.path.join(args.base_file_dir, split, scene))
                save_dir_list.append(os.path.join(args.base_save_dir, split, scene))

    assert len(file_dir_list) == len(save_dir_list)

    for file_dir, save_dir in zip(file_dir_list, save_dir_list):
        assert os.path.exists(file_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        file_name_list = glob.glob(os.path.join(file_dir, '*.aedat4'))
        for file_name in tqdm(file_name_list):

            file_base_name = os.path.basename(file_name)
            # print('\nfile_base_name: ', file_base_name)


            ## Create aedat4 reader
            reader = get_reader(file_name)
            camera_name = reader.getCameraName()
            width, height = reader.getFrameResolution()
            # print('Camera name: {}'.format(camera_name))
            # print('width, height: ', width, height)


            start_time, end_time = reader.getTimeRange()
            duration = end_time - start_time
            # print('start_time, end_time: ', start_time, end_time)
            # print('duration: ', duration)

            image_list = []
            image_timestamp_list = []
            while reader.isRunning():
                frame = reader.getNextFrame()
                if frame is not None:
                    image_list.append(frame.image)
                    image_timestamp_list.append(np.int64(frame.timestamp))

            assert len(image_list) == len(image_timestamp_list)

            frames_dict = {
                'image': np.stack(image_list, axis=0),
                'timestamp': np.stack(image_timestamp_list, axis=0),
                'height': np.int32(height),
                'width': np.int32(width)
            }

            init_time = frames_dict['timestamp'][0]
            frames_dict['timestamp'] = frames_dict['timestamp'] - init_time
            # print(np.sort(frames_dict['timestamp']) == frames_dict['timestamp'])

            if not np.all(frames_dict['timestamp']>=0):
                print('\nfile_base_name: ', file_base_name)

                wrong_idxs = np.where(frames_dict['timestamp']<0)[0]
                for wrong_idx in wrong_idxs:
                    assert  0 < wrong_idx < frames_dict['timestamp'].shape[0]-1
                    if 0 < wrong_idx < frames_dict['timestamp'].shape[0]-1:
                        frames_dict['timestamp'][wrong_idx] = np.int64((frames_dict['timestamp'][wrong_idx-1]+frames_dict['timestamp'][wrong_idx+1])/2)   
                assert np.all(frames_dict['timestamp']>=0)                 

            event_store = reader.getEventsTimeRange(start_time, end_time)
            events, _, _ = EventBuffer.store_to_ndarray(event_store)
            # print('events.shape: ', events.shape)   # [N, 4(x, y, p(0,1), t(microseconds))]

            if events[0, 3] < init_time:
                events = events[events[:, 3]>=init_time]   #

            events_dict = {
                'x': events[:, 0].astype(np.uint16),
                'y': events[:, 1].astype(np.uint16),
                'p': (events[:, 2]*2-1).astype(np.int8),
                't': events[:, 3].astype(np.int64),
                'height': np.int32(height),
                'width': np.int32(width)
            }

            events_dict['t'] = events_dict['t'] - init_time

            h5_file_path = '{}/{}.h5'.format(save_dir, file_base_name)
            h5_file = h5py.File(h5_file_path, 'w')

            h5_event_group = h5_file.create_group('events')
            for k, v in events_dict.items():
                # print(k)
                # print(v.shape)
                # print(v.dtype)
                h5_event_group.create_dataset(name='{}'.format(k), data=v, dtype=v.dtype, chunks=v.shape)

            h5_frame_group = h5_file.create_group('frames')
            for k, v in frames_dict.items():
                # print(k)
                # print(v.shape)
                h5_frame_group.create_dataset(name='{}'.format(k), data=v, dtype=v.dtype, chunks=v.shape)

            h5_frame_group = h5_file.create_group('init_time')
            h5_frame_group.create_dataset(name='time', data=init_time, dtype=np.int64)


            h5_file.close()






















