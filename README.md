# Lightweight LiDAR fusion for Panoptic Segmentation in Adverse Conditions

:2nd_place_medal: **second** place at the [MUSES-AXPS](https://urvis-workshop.github.io/challenge-Muses.html) challenge (**URVIS** @ **CVPRW 2026**)

## Overview
This repository contains the code developed to participate in the **MUSES Adverse-to-eXtreme Panoptic Segmentation (AXPS)** challenge, whose task is to perform panoptic segmentation in adverse illumination and weather conditions leveraging multi-modal data. The competition focuses on the [MUSES dataset](https://github.com/timbroed/MUSES), which contains images, LiDAR, radar and event modalities.

The proposed approach integrates visual and LiDAR features in a lightweight fashion, using an efficient **multi-scale**, **mid-fusion** module. Despite not achieving the SOTA performance of other works, this approach brings a modest improvement in panoptic quality (PQ) using few more parameters and with a minimal latency overhead compared to the corresponding RGB-only Mask2Former model. It also provides better performance than a naive early fusion approach.

This work achieved the **second** place in the challenge and scores 51.7% PQ on the [MUSES panoptic segmentation benchmark](https://www.codabench.org/competitions/13987/).

For more details on the methodology and more insights on the results, you can read the report [here](./report.pdf).

## Installation
This work is developed using:
- `python=3.13`
- `torch=2.10`
- `torchvision=0.25`
- `CUDA=12.8`

To install the required packages, run:

```
pip install -r requirements.txt
```

The evaluation is performed using the [COCO Panoptic API](https://github.com/cocodataset/panopticapi). To install this API, run:

```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Results
The following table summarizes the results on the MUSES `validation` set of the three main approaches:

|Model|Modalities|GFLOPs|Params (M)|Latency (ms)|PQ Things (%)|PQ Stuff (%)|PQ (%)|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Mask2Former|Camera|136|47|185|35.0|57.6|48.1|
|Early fusion|Camera + LiDAR|137|47|186|36.0|57.9|48.7|
|Mid fusion|Camera + LiDAR|144|53|198|**39.2**|**60.2**|**51.4**|

The following table shows the detailed PQ performance for each category on the MUSES `validation` set:

|Model| Clear | Fog |Rain |Snow |Day |Night |All|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Mask2Former|46.1|58.3|43.1|46.2|49.4|44.0|48.1|
|Early fusion|47.2|**59.4**|44.0|44.0|49.0|46.1|48.7|
|Mid fusion|**51.3**|57.6|**48.1**|**50.0**|**50.8**|**51.0**|**51.4**|

The measurements refer to a single `NVIDIA Tesla T4` GPU with 15 GB of available memory.

## Acknowledgements
This work relies in part on the following:
- [MUSES SDK](https://github.com/timbroed/MUSES)
- [Mask2Former](https://github.com/facebookresearch/mask2former)