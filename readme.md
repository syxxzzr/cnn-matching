# Image Matching using CNN feature

## Overview
 
Aiming at the problem that the differences in heterogeneous remote sensing images in imaging modes, time phases, and resolutions make matching difficult, a new deep learning feature matching method is proposed. The results show that the algorithm in this paper has strong adaptability and robustness, and is superior to other algorithms in terms of the number and distribution of matching points, efficiency, and adaptability.
This repository contains the implementation of the following paper: 

```text
"Deep learning algorithm for feature matching of cross modality remote sensing images" （in Chinese）
异源遥感影像特征匹配的深度学习算法
```
[paper pdf](http://xb.sinomaps.com/CN/10.11947/j.AGCS.2021.20200048)

The main idea and code of feature extracting in this repository are based on [D2-Net](https://dusmanu.com/publications/d2-net.html).
 
## Matching result： 
![Image text](https://raw.githubusercontent.com/lan-cz/cnn-matching/master/result/1.jpeg)
Matching result between google earth images （in 2009 & 2018）

![Image text](https://raw.githubusercontent.com/lan-cz/cnn-matching/master/result/2.jpeg)
Matching result between uav optical  image and thermal infrared image

![Image text](https://raw.githubusercontent.com/lan-cz/cnn-matching/master/result/3.jpeg)
Matching result between SAR image （GF-3) & optical satellite(ZY-3)  image

![Image text](https://raw.githubusercontent.com/lan-cz/cnn-matching/master/result/4.jpeg)
Matching result between satellite image & map

## Getting start:
Python 3.7+ is recommended for running our code.

### Dependencies

- PyTorch 1.4.0+
- OpenCV
- SciPy
- Matplotlib
- skimage

### Use Conda
[Conda](https://docs.conda.io/en/latest/) can be used to install the required packages:

### Use UV
[UV](https://uv-vm.org/) can also be used to install the required packages:

```bash
uv sync
```

### Dataset
We collected a set of test data named "df-sm-data", including images from space-borne SAR and visible light sensors, drone thermal infrared sensors, and Google Earth images. You may find them in the directory  "df-sm-data" in this repository.

### Downloading the models

The off-the-shelf **VGG16** weights and their tuned counterpart can be downloaded by running:

```bash
mkdir models
wget https://dusmanu.com/files/d2-net/d2_tf.pth -O models/d2_tf.pth
```

## Usage
`cnnmatching.py` contains the majority of the code. Run `cnnmatching.py` for testing:

```bash
python3 cnnmatching.py
```

or use `UV`:

```bash
uv run python3 cnnmatching.py
```

You may change the images path in the code just like:

```bash
imgfile1 = 'df-ms-data/1/df-googleearth-500-20091227.jpg'
imgfile2 = 'df-ms-data/1/df-googleearth-500-20181029.jpg'
```
