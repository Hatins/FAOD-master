# Creating datasets in Event-RGB Mismatch and Train-Infer Mismatch

## Download Raw datasets

<table>
  <tr>
    <th style="text-align:center;">Download Links</th>
    <th style="text-align:center;"><a href="https://drive.google.com/drive/folders/12PprdOSXhIrlp-xPKeboaVf7G8SPuyJB?usp=drive_link">PKU-DAVIS-SOD</a></td>
    <th style="text-align:center;"><a href="https://drive.google.com/drive/folders/1sqaqS2TWkx8tSdVj4WFJD1uugUaKSX9j?usp=drive_link">DSEC-Detection</a></td>
  </tr>
</table>

## Generate datasets

We provide the scipt for generating PKU-DAVIS-SOD dataset in arbitary frequency combination.

set ``Input_Dir`` and ``Target_Dir`` respectively.

set the ``image_downsampling_rate`` and ``event_upsampling_rate``.

The frequencies of RGB and Event are 25 * image_upsampling_rate and 25 * event_upsampling_rate.

e.g., when you set image_downsampling_rate=0.2, event_upsampling_rate=8, you will get
the dataset with RGB frequency = 5 and Event frequency = 200.

```python
python frame_construction/main.py input_dir={Input_Dir} target_dir={Target_Dir} image_upsampling_rate={image_downsampling_rate} event_upsampling_rate={event_upsampling_rate}
```

## Generate Event frames for DSEC-Detection

We only provide the script for generating paired Event-RGB dataset in DSEC-Detection. But if you want to evaluate the performance of FAOD on Event-RGB Mismatch and Train-Infer Mismatch
DSEC-Detection, you can just follow the code below.

set ``Input_Dir`` and ``Target_Dir`` respectively.

```python
python frame_construction/main_dsec.py input_dir={Input_Dir} target_dir={Target_Dir}
```

