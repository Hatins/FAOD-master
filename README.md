# Frequency-Adaptive Low-Latency Object Detection Using Events and Frames
![License](https://img.shields.io/badge/license-MIT-yellow) ![Language](https://img.shields.io/badge/language-python3.11-brightgreen) ![cuda](http://img.shields.io/badge/cuda-11.8-red)
   

Official code repository for Frequency-Adaptive Low-Latency Object Detection Using Events and Frames.
<p align="center">
  <img src="readme/imgs/framework.png" width="750">
</p>

## Important Notes
:star: The advantages of this repository in dealing object detection using both Events and Frames
- We follow the data format of [RVT](https://github.com/uzh-rpg/RVT), and all datasets are now easier to handle, smaller, and faster to read and write. We appreciate the excellent work of
  [Mr magehrig](https://github.com/magehrig) and the [RPG](https://github.com/uzh-rpg). If you are familiar with the RVT, it will be easy to follow this project.

- Our model is very lightweight, small in size, fast, and can be trained end-to-end on a GPU with 24G of memory.

- We do not perform any additional post-processing (except NMS) during training and testing, and we used all categories of the dataset during training to ensure fair evaluation.

- We provide all the data files, including the files before and after the frame building, as well as the pre-trained model. You can flexibly adjust and add your own design.

## Videos

### Under Event-RGB Mismatch
<p align="center">
  <img src="readme/videos/FAOD_unpaired_2.gif" width="750">
</p>


### Under Train-Infer Mismatch
<p align="center">
  <img src="readme/videos/faod_freq_2.gif" width="750">
</p>

## Installation
We recommend using cuda11.8 to avoid unnecessary environmental problems.
```
conda create -y -n faod python=3.11

conda activate faod

pip install torch==2.1.1 torchvision==0.16.1 torchdata==0.7.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

pip install pandas plotly opencv-python tabulate pycocotools bbox-visualizer StrEnum hydra-core einops torchdata tqdm numba h5py hdf5plugin lovely-tensors tensorboardX pykeops scikit-learn ipdb timm opencv-python-headless wandb==0.14.0 pytorch_lightning==1.8.6 numpy==1.26.3

pip install openmim

mim install mmcv
```
## Required Data
We provide datasets with the similar format of [RVT](https://github.com/uzh-rpg/RVT) for easy implements. 

Noth that the following datasets are paired Event-RGB. Trying to evaluate ``Event-RGB Mismatch`` and ``Train-Infer Mismatch``?
Following these [instructions](https://github.com/Hatins/FAOD-master/blob/main/readme/readme/frame_construcation.md) to create unpaired Event-RGB datasets. 
<table>
  <tr>
    <th style="text-align:center;">Download Links Google Drive</th>
    <th style="text-align:center;"><a href="https://drive.google.com/drive/folders/12PprdOSXhIrlp-xPKeboaVf7G8SPuyJB?usp=drive_link">PKU-DAVIS-SOD</a></td>
    <th style="text-align:center;"><a href="https://drive.google.com/drive/folders/1sqaqS2TWkx8tSdVj4WFJD1uugUaKSX9j?usp=drive_link">DSEC-Detection</a></td>
  </tr>
</table>

<table>
  <tr>
    <th style="text-align:center;">Download Links</th>
    <th style="text-align:center;">PKU-DAVIS-SOD</th>
    <th style="text-align:center;">DSEC-Detection</th>
  </tr>
  <tr>
    <td style="text-align:center;">Google Drive Link</td>
    <td style="text-align:center;">
      <a href="https://drive.google.com/drive/folders/12PprdOSXhIrlp-xPKeboaVf7G8SPuyJB?usp=drive_link">PKU-DAVIS-SOD</a>
    </td>
    <td style="text-align:center;">
      <a href="https://drive.google.com/drive/folders/1sqaqS2TWkx8tSdVj4WFJD1uugUaKSX9j?usp=drive_link">DSEC-Detection</a>
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">OneDrive Link</td>
    <td style="text-align:center;">
      <a href="https://onedrive.live.com/?cid=1234567890abcdef&authkey=%21ABCDEFGHIJKL&folderid=1234567890abcdef">PKU-DAVIS-SOD</a>
    </td>
    <td style="text-align:center;">
      <a href="https://onedrive.live.com/?cid=1234567890abcdef&authkey=%21ABCDEFGHIJKL&folderid=1234567890abcdef">DSEC-Detection</a>
    </td>
  </tr>
</table>
## Checkpoints
<table>
  <tr>
    <th style="text-align:center;">PKU-DAVIS-SOD (Time Shift)</th>
    <th style="text-align:center;">PKU-DAVIS-SOD</th>
    <th style="text-align:center;">DSEC-Detection</th>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/15Xk8fQ0h3zulg0CBtSncB-PIRr6e4e_4/view?usp=drive_link">mAP = 29.7</a></td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1lTzr0X7eXKzeS0wVU8tzg6JYj2cMJCEr/view?usp=drive_link">mAP = 30.5</a></td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/15HqCsKFnRvv1D1dzEmvipLpcCCjtKNEe/view?usp=drive_link">mAP = 42.5</a></td>
  </tr>
</table>

## Validation with pre-trained models
Define the ``DATASET ['pku_fusion', 'dsec']``, ``DATA_PATH``, ``CHECKPOINT``, ``use_test_set [True, False]``, and then run the following command:
```python
python validation.py dataset={DATASET} dataset.path={DATA_PATH} checkpoint={CHECKPOINT} use_test_set={use_test_set} +experiment/{DATASET}='base.yaml'
```
Other settings like ``use_test_set``, ``training.precision``, ``batch_size.eval``, ``hardware.num_workers`` can be set in file ``config/val.yaml`` 
and ``config/experiment/{DATASET}/default.yaml`` conveniently.

## Train FAOD with scratch
Define the ``DATASET``, ``DATA_PATH``, and then run the following command:
```python
python train.py dataset={DATASET} dataset.path={DATA_PATH} +experiment/{DATASET}='base.yaml'
```
Other settings like ``training.precision``, ``batch_size.train``, ``hardware.num_workers`` can be set in file ``config/train.yaml`` 
and ``config/experiment/{DATASET}/default.yaml`` conveniently.
Training FAOD with/without Time Shift? Following this instruction.

## Visualization
The relevant content is in ``demo.py``.

You need to set ``mode = ['pre', 'gt']``, and  ``show_mode = ['event','rgb','mixed']``.

And indicate the sequence you want to visualize, e.g., ``PKU-H5-Process/freq_1_1/test/001_test_low_light``.

Then run the code :

```python
python demo.py dataset={DATASET} dataset.path={DATA_PATH} checkpoint={CHECKPOINT} +experiment/{DATASET}='base.yaml'
```

The results will be saved in ``./gt`` or ``./predictions``. You can also ajust the destination path by yourself.


