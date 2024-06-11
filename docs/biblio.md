# Jolineedle biblio

## Reinforcement learning

### State of the art paper (2021)

https://arxiv.org/pdf/2108.11510.pdf#section.6

- 43: RL on the bbox itself -> not applicable
- 25: hierarchy -> can be interesting
- 251: "Reinforcement Learning for Visual Object Detection" -> see below
- 170: Tree RL with a window you can translate and scale? so you're zooming in the image and moving the window until you find the object? -> not sure how it works
- 246: Seems to be an application of 170 to breast lesion
- 386: uses RL to improve bounding box. Action are operations applied to the bounding box. -> not applicable
- 15: bbox refinement -> not applicable
- 367: "Efficient Object Detection in Large Images Using Deep Reinforcement Learning"

Check: 170, 246, 367

### A Visual Active Search Framework for Geospatial Exploration (2022)

https://arxiv.org/pdf/2211.15788.pdf

Cut image in patches, perform RL on it to identify patches with targets on them.
Action is the number of the patch (the model can jump indefinitely)
The model learns to prefict next patch based on the already visited patches and their labels

### Recurrent Models of Visual Attention (2014)

https://arxiv.org/pdf/1406.6247.pdf

Three different resolution, mimicking peripheral vision. Trajectory by coordinates on an image.
This is primarily a classification model.

### Dynamic Zoom-in Network for Fast Object Detection in Large Images (2017)

https://arxiv.org/abs/1711.05187

2 networks:
- initial accuracy gain regression network, as the state of the reinforcement algo
- zoom-in Q function learning network: the reinforcement network. Action is the bounding box of the zoom, reward is roughly (predictions gained from coarse detection - size of the zone relatively to the whole image) (the higher the reward, the better)

"Coarse-to-fine": The idea is to detect regions of interest in big images ("coarse" level) and perform "fine" detection only on some regions to save resources

### Reinforcement Learning for Visual Object Detection (2016)

https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Mathe_Reinforcement_Learning_for_CVPR_2016_paper.html

Summary: article from 2016, works like RCNN, performance gains not transferable to current models.

Performs RL on the proposals (provided by a segmentation algorithm).

The algorithm looks at multiple places, then gives its final prediction ("done" action).

Which part of the image is evaluated at each step? -> Proposal ROI resized for the network.

### Efficient Object Detection in Large Images Using Deep Reinforcement Learning (2020)

https://openaccess.thecvf.com/content_WACV_2020/papers/Uzkent_Efficient_Object_Detection_in_Large_Images_Using_Deep_Reinforcement_Learning_WACV_2020_paper.pdf

Coarse Level Search: select patches to perform fine level search
Fine Level Search: same thing on selected patches

## Coarse to fine

Use of downsampled large image / lightweight network on HD image to filter out zones with few objects to detect

### A Coarse to Fine Framework for Object Detection in High Resolution Image (2023)

https://arxiv.org/pdf/2303.01219.pdf

- 1 detector on downsampled image
- 1 lightweight detector on HD image
- 1 detector on regions identified by the first two detectors

Method without RL, the "lightweight" detector determines which regions should be analyzed by the "fine detector".

### A coarse to fine network for fast and accurate object detection in high-resolution images

https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.12042 

Detection with YOLOv3 on image patches. Filter out patch without images at intermediate level in Yolov3 architecture

### PENet: Object Detection using Points Estimation in Aerial Images (2020)

https://arxiv.org/abs/2001.08247

Non maximum merge (NMM) algorithm for clusters at coarse level

### Focus-and-Detect: A Small Object Detection Framework for Aerial Images (2022)

https://arxiv.org/abs/2203.12976

Gaussian mixture models

### LiteEval: A Coarse-to-Fine Framework for Resource Efficient Video Recognition (2019)

https://arxiv.org/abs/1912.01601

- coarse LSTM and fine LSTM, a gate decides whether the fine network should be call, otherwise it's not called
- if frame is skipped for fine network the state of the lstm is copied from the coarse network
- learning gate is a RL problem

