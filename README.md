#"You Only Look Once" - YOLO object detection

## Overview

YOLO algorithm uses convolutional neural networks (CNN) to detect objects with only a single forward propagation through a neural network. This repository creates a model that can detect objects in both images and videos.

## Requirements:

* Python 3.7
* OpenCV 4.5
* Tensorflow 2.4
* Keras 2.4 

## Details:

The program detects the objects from list of 80 Common Objects in Context (COCO) Classes, loaded from a textfile.
Weights of pretrained model have to be downloaded from [here](https://pjreddie.com/media/files/yolov3.weights) and saved in the project folder, then convert it to keras h5 file by running yad2k.py:

```python
python yad2k.py cfg\yolo.cfg yolov3.weights data\yolo.h5
```

After that the program can be run by:

```python
python yolo_method.py
```

## Example result:

Result after running the script for object threshold = 0.4 and box threshold = 0.5.

<img width="500" height="330" src="/images/res/people2.jpg"/>

## Sources:

* [Python for Computer Vision with OpenCV and Deep Learning course](https://www.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/)
* [Redmon Joseph and Farhadi Ali. YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
* YOLO version 3 model implementation: [Xiaochus](https://github.com/xiaochus/YOLOv3)
