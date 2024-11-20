'''
This script is used to transfer the json format in annotations of SODFormer to the npy format, which is often used in recent works.
the struction of the outcomes
shape->(N,) N denotes the number of the bbox
each bbox-> (t, x, y, w, h, class_id) -> (uint64, float32, float32, float32, float32, uint8)
Author: Hatins
Time: 2024/03/12
The output will be added at the h5_input_path.
We move the labels which have not the corresponding aps images
'''

import numpy as np
import os
import json
import h5py
import hdf5plugin
import argparse 
from tqdm import tqdm

pku_scenes = ['normal', 'low_light', 'motion_blur']
pku_splits = ['train', 'val', 'test']

def get_number_in_str(file_name):
    numeric_part = ''.join(char for char in file_name if char.isnumeric())
    return int(numeric_part)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--josn_input_path', type = str, default='/data2/zht/fusion_detection/PKU-DATA/annotations',help='josn label path')
    parser.add_argument('--h5_input_path', type = str, default='/data2/zht/fusion_detection/PKU-H5',help='raw event path')
    args = parser.parse_args()

    delete_num = 0
    for mode in pku_splits:
        each_mode_josn_dict_path = args.josn_input_path + '/' + mode
        each_mode_h5_dict_path = args.h5_input_path + '/' + mode
        for scene in pku_scenes:
            each_mode_scene_josn_dict_path = each_mode_josn_dict_path + '/' + scene
            each_mode_scene_h5_dict_path = each_mode_h5_dict_path + '/' + scene
            print('prcessing the '+ scene + ' scene in ' + mode)
            for filename in tqdm(os.listdir(each_mode_scene_josn_dict_path)):
                josn_file_path = os.path.join(each_mode_scene_josn_dict_path, filename)
                h5_file_path = each_mode_scene_h5_dict_path + '/' + filename + '.aedat4.h5'
                save_path = each_mode_scene_h5_dict_path + '/' + filename
                file_list = os.listdir(josn_file_path)
                # npy_file = np.load(npy_path)
                h5_file = h5py.File(h5_file_path)

                frames_info = h5_file['frames']
                timestamp_info = frames_info['timestamp']

                annotations_list = []
                for josn_file_name in file_list:
                    file_path = os.path.join(josn_file_path, josn_file_name)
                    
                    with open(file_path, 'r') as file:
                        json_data = json.load(file)
                        bbox_num = len(json_data['shapes'])
                        image_path = json_data['imagePath']
                        image_index = get_number_in_str(image_path)

                        if image_index > len(timestamp_info):
                            os.remove(file_path)
                            print(file_path + 'has been moved due to it has not the corresponding aps frame.')
                            delete_num += 1
                            continue

                        timestamp = timestamp_info[image_index-1]

                        for bbox_index in range(bbox_num):
                            x1, y1 = json_data['shapes'][bbox_index]['points'][0]
                            x2, y2 = json_data['shapes'][bbox_index]['points'][1]
                            x_tl = min(x1, x2)
                            y_tl = min(y1, y2)
                            width = abs(x2 - x1)
                            height = abs(y2 - y1)

                            if width == 0 or height == 0:
                                print(file_path, ' has the width or height = 0')

                            class_id = json_data['shapes'][bbox_index]['label']
                            annotation = (timestamp, x_tl, y_tl, width, height, class_id)
                            annotations_list.append(annotation)


                dtype = [('t', np.uint64), ('x', np.float32), ('y', np.float32), ('w', np.float32), ('h', np.float32),  ('class_id', np.uint8)]

                annotations_list_numpy = np.array(annotations_list, dtype = dtype)
                sorted_data_array = np.sort(annotations_list_numpy, order='t')

                save_path = save_path + '_bbox'
                np.save(save_path, sorted_data_array)
    print('delete {} wrong labels'.format(delete_num))


