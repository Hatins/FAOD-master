'''
    This file is used to visualize the dataset after constructing the events to frames.
    Author: Hatins
    Time: 2024/03/13
    We do not visualize on wins directly since it's often not convenient for the user of servers. 
    The results are saved in visualization/outputs/dataset_vis_outputs.
'''

import h5py
import hdf5plugin
import os
from tqdm import tqdm
import numpy as np
import cv2
import argparse
import shutil
import ipdb

def sort_key(filename):
    return int(filename.split('.')[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_file', type = str, default='/data2/zht/fusion_detection/PKU-H5-Process/freq_0.2/test/001_test_low_light/',help='squence name')
    parser.add_argument('--generate_video', type = bool, default=True)
    parser.add_argument('--modality', type = str, default='both', choices=['aps','event','both'])
    parser
    args = parser.parse_args()

    base_path = args.vis_file
    if args.modality == 'event':
        images_out_put_dir = 'visualization/outputs/dataset_vis_outputs/EVENT/images'
        video_out_put_dir = 'visualization/outputs/dataset_vis_outputs/EVENT/video'
    elif args.modality == 'aps':
        images_out_put_dir = 'visualization/outputs/dataset_vis_outputs/APS/images'
        video_out_put_dir = 'visualization/outputs/dataset_vis_outputs/APS/video'
    elif args.modality == 'both':
        images_out_put_dir = 'visualization/outputs/dataset_vis_outputs/BOTH/images'
        video_out_put_dir = 'visualization/outputs/dataset_vis_outputs/BOTH/video'

    if not os.path.exists(images_out_put_dir):
        os.makedirs(images_out_put_dir)
    else:
        shutil.rmtree(images_out_put_dir)
        os.makedirs(images_out_put_dir)


    if not os.path.exists(video_out_put_dir):
        os.makedirs(video_out_put_dir)
    else:
        shutil.rmtree(video_out_put_dir)
        os.makedirs(video_out_put_dir)

    event_frame_path = base_path + 'event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations.h5'
    aps_frame_path = base_path + 'labels_v2/images.h5'
    frame_2_event_path = base_path + 'event_representations_v2/stacked_histogram_dt=50_nbins=10/objframe_idx_2_repr_idx.npy'
    labels_path = base_path + 'labels_v2/labels.npz'

    bbox_color = (0, 255, 0)

    frame_2_event = np.load(frame_2_event_path)

    #including label bboxs and the objframe_idx_2_label_idx
    labels = np.load(labels_path)
    labels_bboxs = labels['labels']
    frame_2_label = labels['objframe_idx_2_label_idx']
    event_frame = h5py.File(event_frame_path)['data']
    aps_frame = h5py.File(aps_frame_path)['data']

    event_frame = np.asarray(event_frame)
    if args.modality == 'event' or args.modality == 'aps':
        width = event_frame.shape[3]
        height = event_frame.shape[2]
    elif args.modality == 'both':
        width = event_frame.shape[3] * 2
        height = event_frame.shape[2]

    num_frame = event_frame.shape[0]


    for frame_index in tqdm(range(num_frame)):

        if frame_index in frame_2_event:
            adding_label = True
            label_index = np.where(frame_index == frame_2_event)[0][0]
            if frame_index + 1 < len(frame_2_label):
                labels_of_frame = labels_bboxs[frame_2_label[frame_index] : frame_2_label[frame_index + 1]]
            else:
                labels_of_frame = labels_bboxs[frame_2_label[frame_index]:]
        else:
            adding_label = False

        if args.modality == 'event':
            single_frame = event_frame[frame_index]
            single_frame_shown = (single_frame.sum(axis=0) * 60).astype(np.uint8)
            single_frame_shown = cv2.cvtColor(single_frame_shown, cv2.COLOR_GRAY2BGR)
        elif args.modality == 'aps':
            single_frame = aps_frame[frame_index]
            single_frame_shown = single_frame.astype(np.uint8)
        elif args.modality == 'both':
            event_single_frame = event_frame[frame_index]
            event_single_frame_shown = (event_single_frame.sum(axis=0) * 60).astype(np.uint8)
            event_single_frame_shown = cv2.cvtColor(event_single_frame_shown, cv2.COLOR_GRAY2BGR)

            aps_single_frame = aps_frame[frame_index]
            aps_single_frame_shown = aps_single_frame.astype(np.uint8)

            unit_frame_shown = [event_single_frame_shown, aps_single_frame_shown]

        if adding_label == True:
            if args.modality == 'event' or args.modality == 'aps':
                for each_bbox in labels_of_frame:
                    x, y, w, h, label = list(each_bbox)[1:6]
                
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    thickness = 2
                    cv2.rectangle(single_frame_shown, (x1, y1), (x2, y2), bbox_color, thickness)
                    cv2.putText(single_frame_shown, str(label), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, thickness)
            elif args.modality == 'both':
                frame_list = []
                for single_frame_shown in unit_frame_shown:
                    for each_bbox in labels_of_frame:
                        x, y, w, h, label = list(each_bbox)[1:6]
                    
                        x1, y1 = x, y
                        x2, y2 = x + w, y + h
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        thickness = 2
                        cv2.rectangle(single_frame_shown, (x1, y1), (x2, y2), bbox_color, thickness)
                        cv2.putText(single_frame_shown, str(label), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, thickness)
                    frame_list.append(single_frame_shown)
                assert frame_list[0].shape == frame_list[1].shape
                single_frame_shown = cv2.hconcat([frame_list[0], frame_list[1]])
        else:
            if args.modality == 'both':
                single_frame_shown = cv2.hconcat([unit_frame_shown[0], unit_frame_shown[1]])

        output_file_path = images_out_put_dir + '/{}.png'.format(frame_index)

        cv2.imwrite(output_file_path, single_frame_shown)

    if args.generate_video == True:
        images = [img for img in os.listdir(images_out_put_dir) if img.endswith(".png")]
        images.sort(key=sort_key)
        video_name = video_out_put_dir + '/output_video.avi'
        fps = int(len(images) / 30) 
        frame_size = (width, height) 
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        video_writer = cv2.VideoWriter(video_name, fourcc, fps, frame_size)  

        for image in images:
            image_path = os.path.join(images_out_put_dir, image)
            frame = cv2.imread(image_path)
            video_writer.write(frame)

        video_writer.release()
        print('Save the video in {}'.format(video_name))





