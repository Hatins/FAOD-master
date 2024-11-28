# Frequency-Adaptive Low-Latency Object Detection Using Events and Frames
<p align="center">
  <img src="readme/imgs/framework.png" width="750">
</p>

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

pip install wandb pandas plotly opencv-python tabulate pycocotools bbox-visualizer StrEnum hydra-core einops torchdata tqdm numba h5py hdf5plugin lovely-tensors tensorboardX pykeops scikit-learn ipdb timm opencv-python-headless pytorch_lightning==1.8.6 numpy==1.26.3

pip install openmim

mim install mmcv
```
## Required Data
<table>
  <tr>
    <th style="text-align:center;"> </th>
    <th style="text-align:center;"><a href="https://1drv.ms/u/c/93289205239bc375/EQue4dcG4M9Ggbu5dM-iOc0Bphskqnh1zua2rogpYNkANw?e=crXrjv">Davis-PKU-Fusion</a></td>
    <th style="text-align:center;"><a href="https://1drv.ms/u/c/93289205239bc375/ETetOpGDDyJDsN_5lTkvdwEBqEvm9kw2aqdXDNCiHn4FAg?e=c1yTGf">DSEC-Fusion</a></td>
  </tr>
</table>

