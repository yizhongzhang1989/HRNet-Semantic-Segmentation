# HRNet for panoramas

## colorMap

在HRNet根目录下的`colorMap.txt`定义了关于各类别的信息。每一行是一类，格式如下：
```
类别名称 数字标号 R G B
```
数字标号不一定是连续的，比如目前只有17类，但是最大的标号是40。在处理数据的过程中，会自动将不连续的标号转换成连续的标号，从第0行开始，这一类在第几行就会被转换成几。只要用VS Code在整个目录里搜索`colorMap.txt`，就能找到所有使用了这一转换过程的地方，以后需要修改的时候只要在这些地方修改就可以了。
由于接下来提到的脚本基本上都会读取这个`colorMap.txt`，所以这些脚本请在HRNet根目录下运行。

另外，各个脚本都是使用opencv读取的数据，所以读入的都是BGR图片，脚本中已经包含了输入时将BGR转为RGB的代码和输出时将RGB转为BGR的代码，所以不需要额外处理，如果想要写其他脚本，建议使用opencv读取数据以保持一致性。另外，opencv读取图片时路径里不能包含中文，所以处理数据集的时候超市的名字都应该改为英文。

## 准备数据

panorama数据集的目录结构如下
````bash
$DATASET_ROOT/
├── store1
│   ├── 1.jpg
│   └── 1.png
├── store3
│   ├── a.jpg
│   └── a.png
└── store4
    ├── def.jpg
    └── def.png
````

在`tools/preprocessPanorama.py`的开头将`root_dir`设为`$DATASET_ROOT`；将`threads`设为想要并行处理的线程数；将`stores`设为想要处理的超市的文件夹名称组成的tuple或者list；将`image_fov`设为x轴方向上的想要的fov；将`image_per_panorama`设为每张panorama切割出的图片数量；将`image_w`和`image_h`设为想要的分辨率。运行该脚本即开始处理数据。

当fov为50时，`image_per_panorama`推荐为10；fov为60时推荐为9；fov为70时推荐为8。这样，相邻两张image的视野重叠大概正好是一半。这个切割的过程会将360°均分为`image_per_panorama`份，然后从一个随机的角度开始，每隔`360°/image_per_panorama`(取整)切出一张图。

处理好的数据将位于`$DATASET_ROOT/merge/image`和`$DATASET_ROOT/merge/label`中。文件名的格式为`超市名称_原来的文件名_切割时的theta值_切割时的fov.jpg`和`超市名称_原来的文件名_切割时的theta值_切割时的fov.png`

然后在`tools/generate_list.py`的开头将`data_dir`设为`$DATASET_ROOT/merge/label`，就将按照4:1:1的比例随机产生`train_list.txt`, `test_list.txt`, `val_list.txt`。在此过程中，会检查每一张label图片，如果一张图片内像素值为255(意味着不属于任何已定义的类别)的像素比例大于百分之一，这张label和对应的image将不会包含在这三个list中。

## 训练

目录结构如下
````bash
$SEG_ROOT/data
└── panorama
    ├── image
    │   ├── 1.jpg
    │   └── 2.jpg
    ├── label
    │   ├── 1.png
    │   └── 2.png
    ├── train_list.txt
    ├── val_list.txt
    └── test_list.txt
````

当使用`experiments/panorama/train.yaml`作为配置文件时，对应的训练好的imagenet的权重应该位于`pretrained_models/`中。然后使用`python -m torch.distributed.launch --nproc_per_node=1 tools/train.py --cfg=experiments/panorama/train.yaml`。

如果是Windows系统的pytorch，因为Windows版的pytorch不自带distributed模块，可以使用`python tools/windows_train.py --cfg=experiments/panorama/train.yaml`来进行训练，这个脚本不支持多gpu训练。如果一定要在Windows上进行多gpu训练，需要从源码编译pytorch，并选上distributed模块。

## 一些工具脚本

### tools/analysisData.py

在这个脚本开头处设置`data_dir`(也就是数据集的`image`和`label`文件夹所在的文件夹)。

这个脚本会统计数据集的分布，最终输出很多行数据，每一行的构成如下：
```
类别名 数字标号 验证集中每张图片该类的平均像素数 测试集中每张图片该类的平均像素数 训练集中每张图片该类的平均像素数 验证集中每张图片该类的方差 测试集中每张图片该类的方差 训练集中每张图片该类的方差数
```

### tools/analysisResult.py

在这个脚本开头处设置`data_dir`(也就是数据集的`image`和`label`文件夹所在的文件夹)，`result_dir`（也就是网络输出的predict图片所在的文件夹），`output_dir`（输出分析结果的文件夹，将在这个文件夹里自动检测并创建`easy`和`hard`文件夹），`topK`（也就是输出的图像数量）

这个脚本会统计每一张predict上像素判断正确率，并且将正确率最高的topK张图片和结果存放在`easy`文件夹中，将正确率最低的topK张图片和结果存放在`hard`文件夹中。

### tools/analysisVideo.py

在命令行使用这个脚本的时候，使用`--cfg`指定定义网络结构的`.yaml`文件；使用`--pth`指定权重文件；使用`--input_video`指定需要分析的视频文件;使用`--output_video`指定输出文件(带路径)，目前只支持输出avi格式的视频；使用`--batch_size`指定跑一次网络处理多少帧视频；使用`--scale_factor`指定将原视频的尺寸放缩多少倍作为网络输入，比如0.5就是长宽各变为原来的一半再输入网络。

最后会输出一个视频，左上角是原始内容，右上角是分割结果，左下角是每个像素点softmax以后分为当前结果类的概率（纯蓝代表0，纯红代表1），右下角是原始内容和分割结果的叠加。

### tools/pipeline.py

这个脚本是使用`utils/inference.py`的一个样例。`utils/inference.py`里面定义了以下三个用来推理的函数：

`single_image_inference(network, image)`这个函数读入一个网络和一张已经预处理好的图片，可以返回一张predict图片

`batch_inference(network, batch, output_probability=False)`这个函数读入一个网络和一个装有同样大小的预处理好的图片的tuple或者list，返回一个3维的numpy数组，即一个batch的predict图片。如果`output_probability=True`，那么还会额外返回一个batch的probability，即每一个像素都是一个0-1之间的概率值

`inference(network, arguments)`这个函数读入一个网络和一个dict，arguments["input_list"]是一组输入图片的路径组成的tuple或者list；arguments["output_list"]是一组输出图片的路径组成的tuple或者list；arguments["preprocess_function"]定义了一个预处理函数，是可选项，如果dict中没有这一项会使用`utils/inference.py`中定义的默认预处理函数。

# High-resolution networks (HRNets) for Semantic Segmentation 

## Branches
- This is the implementation for PyTroch 1.1.
- The HRNet + OCR version ia available [here](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR).
- The PyTroch 0.4.1 version is available [here](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/master).

## News
- [2020/03/13] Our paper is accepted by TPAMI: [Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/pdf/1908.07919.pdf).
- HRNet + OCR + SegFix: Rank \#1 (84.5) in [Cityscapes leaderboard](https://www.cityscapes-dataset.com/benchmarks/). OCR: object contextual represenations [pdf](https://arxiv.org/pdf/1909.11065.pdf). ***HRNet + OCR is reproduced [here](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR)***.
- Thanks Google and UIUC researchers. A modified HRNet combined with semantic and instance multi-scale context achieves SOTA panoptic segmentation result on the Mapillary Vista challenge. See [the paper](https://arxiv.org/pdf/1910.04751.pdf).
- Small HRNet models for Cityscapes segmentation. Superior to MobileNetV2Plus ....
- Rank \#1 (83.7) in [Cityscapes leaderboard](https://www.cityscapes-dataset.com/benchmarks/). HRNet combined with an extension of [object context](https://arxiv.org/pdf/1809.00916.pdf)

- Pytorch-v1.1 and the official Sync-BN supported. We have reproduced the cityscapes results on the new codebase. Please check the [pytorch-v1.1 branch](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1).

## Introduction
This is the official code of [high-resolution representations for Semantic Segmentation](https://arxiv.org/abs/1904.04514). 
We augment the HRNet with a very simple segmentation head shown in the figure below. We aggregate the output representations at four different resolutions, and then use a 1x1 convolutions to fuse these representations. The output representations is fed into the classifier. We evaluate our methods on three datasets, Cityscapes, PASCAL-Context and LIP.

![](figures/seg-hrnet.png)

## Segmentation models
HRNetV2 Segmentation models are now available. All the results are reproduced by using this repo!!!

The models are initialized by the weights pretrained on the ImageNet. You can download the pretrained models from  https://github.com/HRNet/HRNet-Image-Classification.

### Memory usage and time cost
Memory and time cost comparison for semantic segmentation on PyTorch 1.0 in terms of training/inference memory and training/inference time. The numbers for training are obtained on a machine with 4 V100 GPU cards. During training, the input size is 512x1024 and the batch size is 8. The numbers for inference are obtained on a single V100 GPU card. The input size is 1024x2048.

| approach | train mem | train sec./iter |infer. mem | infer sec./image | mIoU |
| :--: | :--: | :--: | :--: | :--: | :--: | 
| PSPNet | 14.4G | 0.837| 1.60G | 0.397 | 79.7 | 
| DeepLabV3 | 13.3G | 0.850 | 1.15G | 0.411 | 78.5 | 
| HRNet-W48 | 13.9G | 0.692 | 1.79G | 0.150 | 81.1 | 


### Big models

1. Performance on the Cityscapes dataset. The models are trained and tested with the input size of 512x1024 and 1024x2048 respectively.
If multi-scale testing is used, we adopt scales: 0.5,0.75,1.0,1.25,1.5,1.75.

| model | Train Set | Test Set |#Params | GFLOPs | OHEM | Multi-scale| Flip | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| HRNetV2-W48 | Train | Val | 65.8M | 696.2 | No | No | No | 81.1 | [OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33gSlK7Fju_sXCxFUt?e=WZ96Ck)/[BaiduYun(Access Code:t6ri)](https://pan.baidu.com/s/1GXNPm5_DuzVVoKob2pZguA)|

2. Performance on the LIP dataset. The models are trained and tested with the input size of 473x473.

| model |#Params | GFLOPs | OHEM | Multi-scale| Flip | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| HRNetV2-W48 | 65.8M | 74.3 | No | No | Yes | 55.8 | [OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33gSjZUHtqfojPfBc6?e=4sE90v)/[BaiduYun(Access Code:sbgy)](https://pan.baidu.com/s/17LAPB-7wsGFPVpHF51tI-w)|

### Small models

The models are initialized by the weights pretrained on the ImageNet. You can download the pretrained models from  https://github.com/HRNet/HRNet-Image-Classification.

Performance on the Cityscapes dataset. The models are trained and tested with the input size of 512x1024 and 1024x2048 respectively. The results of other small models are obtained from Structured Knowledge Distillation for Semantic Segmentation(https://arxiv.org/abs/1903.04197).

| model | Train Set | Test Set |#Params | GFLOPs | OHEM | Multi-scale| Flip | Distillation | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| SQ | Train | Val | - | - | No | No | No | No | 59.8 | |
| CRF-RNN | Train | Val | - | - | No | No | No | No | 62.5 | |
| Dilation10 | Train | Val | 140.8 | - | No | No | No | No | 67.1 | |
| ICNet | Train | Val | - | - | No | No | No | No | 70.6 | |
| ResNet18(1.0) | Train | Val | 15.2 | 477.6 | No | No | No | No | 69.1 | |
| ResNet18(1.0) | Train | Val | 15.2 | 477.6 | No | No | No | Yes | 72.7 | |
| MD(Enhanced) | Train | Val | 14.4 | 240.2 | No | No | No | No | 67.3 | |
| MD(Enhanced) | Train | Val | 14.4 | 240.2 | No | No | No | Yes | 71.9 | |
| MobileNetV2Plus | Train | Val | 8.3 | 320.9 | No | No | No | No | 70.1 | |
| MobileNetV2Plus | Train | Val | 8.3 | 320.9 | No | No | No | Yes | 74.5 | |
| HRNetV2-W18-Small-v1 | Train | Val | 1.5M | 31.1 | No | No | No | No | 70.3 | [OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33gSEsg-2sxTmZL2AT?e=AqHbjh)/[BaiduYun(Access Code:63be)](https://pan.baidu.com/s/17pr-he0HEBycHtUdfqWr3g)|
| HRNetV2-W18-Small-v2 | Train | Val | 3.9M | 71.6 | No | No | No | No | 76.2 | [OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33gSAL4OurOW0RX4JH?e=ptLwpW)/[BaiduYun(Access Code:k23v)](https://pan.baidu.com/s/155Qxztpc-DU_zmrSOUvS5Q)|

## Quick start
### Install
1. Install PyTorch=1.1.0 following the [official instructions](https://pytorch.org/)
2. git clone https://github.com/HRNet/HRNet-Semantic-Segmentation $SEG_ROOT
3. Install dependencies: pip install -r requirements.txt

If you want to train and evaluate our models on PASCAL-Context, you need to install [details](https://github.com/zhanghang1989/detail-api).
````bash
# PASCAL_CTX=/path/to/PASCAL-Context/
git clone https://github.com/zhanghang1989/detail-api.git $PASCAL_CTX
cd $PASCAL_CTX/PythonAPI
python setup.py install
````

### Data preparation
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/), [LIP](http://sysu-hcp.net/lip/) and [PASCAL-Context](https://cs.stanford.edu/~roozbeh/pascal-context/) datasets.

Your directory tree should be look like this:
````bash
$SEG_ROOT/data
├── cityscapes
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── leftImg8bit
│       ├── test
│       ├── train
│       └── val
├── lip
│   ├── TrainVal_images
│   │   ├── train_images
│   │   └── val_images
│   └── TrainVal_parsing_annotations
│       ├── train_segmentations
│       ├── train_segmentations_reversed
│       └── val_segmentations
├── pascal_ctx
│   ├── common
│   ├── PythonAPI
│   ├── res
│   └── VOCdevkit
│       └── VOC2010
├── list
│   ├── cityscapes
│   │   ├── test.lst
│   │   ├── trainval.lst
│   │   └── val.lst
│   ├── lip
│   │   ├── testvalList.txt
│   │   ├── trainList.txt
│   │   └── valList.txt
````

### Train and test
Please specify the configuration file.

For example, train the HRNet-W48 on Cityscapes with a batch size of 12 on 4 GPUs:
````bash
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
````

For example, evaluating our model on the Cityscapes validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     TEST.MODEL_FILE hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
````
Evaluating our model on the Cityscapes test set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     DATASET.TEST_SET list/cityscapes/test.lst \
                     TEST.MODEL_FILE hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
````
Evaluating our model on the PASCAL-Context validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/pascal_ctx/seg_hrnet_w48_cls59_480x480_sgd_lr4e-3_wd1e-4_bs_16_epoch200.yaml \
                     DATASET.TEST_SET testval \
                     TEST.MODEL_FILE hrnet_w48_pascal_context_cls59_480x480.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
                     TEST.FLIP_TEST True
````
Evaluating our model on the LIP validation set with flip testing:
````bash
python tools/test.py --cfg experiments/lip/seg_hrnet_w48_473x473_sgd_lr7e-3_wd5e-4_bs_40_epoch150.yaml \
                     DATASET.TEST_SET list/lip/testvalList.txt \
                     TEST.MODEL_FILE hrnet_w48_lip_cls20_473x473.pth \
                     TEST.FLIP_TEST True \
                     TEST.NUM_SAMPLES 0
````

## Other applications of HRNet
* [Human pose estimation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
* [Image Classification](https://github.com/HRNet/HRNet-Image-Classification)
* [Object detection](https://github.com/HRNet/HRNet-Object-Detection)
* [Facial landmark detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)

## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI}
  year={2019}
}
````

## Reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, 
    Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI.  [download](https://arxiv.org/pdf/1908.07919.pdf)

## Acknowledgement
We adopt ~~sync-bn implemented by [InplaceABN](https://github.com/mapillary/inplace_abn).~~ the PyTorch official syncbn.

We adopt data precosessing on the PASCAL-Context dataset, implemented by [PASCAL API](https://github.com/zhanghang1989/detail-api).
