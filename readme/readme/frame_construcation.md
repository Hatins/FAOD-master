# Creating datasets in Event-RGB Mismatch and Train-Infer Mismatch

## Download Raw datasets

<table>
  <tr>
    <th style="text-align:center;">Download Links</th>
    <th style="text-align:center;"><a href="https://entuedu-my.sharepoint.com/:u:/g/personal/haitian003_e_ntu_edu_sg/IQDTjDVcu0zQSI9HNjywl6JmAYVHuAYRxHd9WgbwR5iVsEs?e=ygYCg3">PKU-DAVIS-SOD</a></td>
    <th style="text-align:center;"><a href="https://entuedu-my.sharepoint.com/:u:/g/personal/haitian003_e_ntu_edu_sg/IQDe2pAwCEC6TaC7-IaWZIPtARC_kIoeJOap7q67AMhtScM?e=WppzEu">DSEC-Detection</a></td>
  </tr>
</table>

## Generate datasets for PKU-DAVIS-SOD

We provide the scipt for generating PKU-DAVIS-SOD dataset in arbitary frequency combination.

set ``Input_Dir`` and ``Target_Dir`` respectively.

set the ``image_downsampling_rate`` and ``event_upsampling_rate``.

The frequencies of RGB and Event are 25 * image_upsampling_rate and 25 * event_upsampling_rate.

e.g., when you set image_downsampling_rate=0.2, event_upsampling_rate=8, you will get
the dataset with RGB frequency = 5 and Event frequency = 200.

```python
python frame_construction/main.py --input_dir={Input_Dir} --target_dir={Target_Dir} --image_upsampling_rate={image_downsampling_rate} --event_upsampling_rate={event_upsampling_rate}
```

## Generate datasets for DSEC-Detection

We provide the scipt for generating DSEC-Detection dataset in arbitary frequency combination.

set ``Input_Dir`` and ``Target_Dir`` respectively.

set the ``image_downsampling_rate`` and ``event_upsampling_rate``.

The frequencies of RGB and Event are 20 * image_upsampling_rate and 20 * event_upsampling_rate.

e.g., when you set image_downsampling_rate=0.2, event_upsampling_rate=8, you will get
the dataset with RGB frequency = 4 and Event frequency = 160.

```python
python frame_construction/main_dsec.py --input_dir={Input_Dir} --target_dir={Target_Dir} --image_upsampling_rate={image_downsampling_rate} --event_upsampling_rate={event_upsampling_rate}
```

