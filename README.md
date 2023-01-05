# [Towards Classification of Architectural Styles of Traditional Settlements using DL: A Dataset, a New Framework, and Its Interpretability](https://www.mdpi.com/2072-4292/14/20/5250)

[Paper](https://www.mdpi.com/2072-4292/14/20/5250)
Authors: Qing HAN, Chao YIN*, Yunyuan DENG, Peilin LIU

Classification of architectural style for Chinese traditional settlements (CTSs) has become a crucial task for developing and preserving settlements. Traditionally, the classification of CTSs primarily relies on manual work, which is inefficient and time-consuming. Inspired by the tremendous success of deep learning (DL), some recent studies attempted to apply DL networks such as convolution neural networks (CNNs) to achieve automated classification of the architecture styles. However, these studies suffer overfitting problems of the CNNs, leading to inferior classification performance. Moreover, most of the studies apply the CNNs as a black box providing limited interpretability. To address these limitations, a new DL classification framework is proposed to overcome the overfitting problem by transfer learning and learning-based data augmentation technique (i.e., AutoAugment). Furthermore, we also employ class activation map (CAM) visualization technique to help understand how the CNN classifiers work to abstract patterns from the input.

Overview of the framework:
![Overview](images/fig1.jpg)

## Requirements

We recommend using python3 and ubuntu 18.04+. Create a conda environment with the following commands:

```
conda create –n CTS python=3.6
source activate CTS
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip3 install scikit-learn matplotlib pandas seaborn pyyaml
conda install -c conda-forge opencv
pip3 install -r requirements.txt
```

## Clone the repo and download the dataset

- Clone this repo with `git clone https://github.com/PointCloudYC/CTS.git`

- Download the ArchiSytle dataset from [ArchiStyle dataset](https://hkustconnect-my.sharepoint.com/:u:/g/personal/cyinac_connect_ust_hk/EW357p4zW0JKoadv5Ywcp7oBFBZ63RKSpjeRBXFokeIm-A?e=AQiD5u), then unzip it to the `data` folder. The resulting file structure should look like:

```
<root>
├── data
│   └── 256x256_ArchiStyle
│       └── train
│           ├── 0_2.jpg
│           └── ...
│       └── val
│           ├── 0_1.jpg
│           └── ...
│       └── test
│           ├── 0_72.jpg
│           └── ...
│   └── ArchiStyle
├── ...
```

## Task

Given an image of a Traditional Style (TS) image, predict the correct style, e.g., Su, Jing style, etc..

## Train, evaluate and predict

```
# baseline
python function/train.py --model_dir ../experiments/base_model/
python function/evaluate.py --model_dir ../experiments/base_model/

# data aug.
python function/train.py --model_dir ../experiments/data_aug/
python function/evaluate.py --model_dir ../experiments/data_aug/

# hyper-tuning
# python function/search_hyperparams.py 

# select baseline models
# python function/search_models.py 

# synthesize results
# python function/synthesize_results.py
```

see the `run.sh`

## Citation

@article{CTS,
    Author = {Qing HAN, Chao YIN*, Yunyuan DENG, Peilin LIU},
    Title = {Towards Classification of Architectural Styles of Chinese Traditional Settlements using Deep Learning: A Dataset, a New Framework, and Its Interpretability},
    Journal = {Remote Sensing},
    Year = {2022}
   }