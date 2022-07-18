# Jittor 可微渲染新视角生成比赛 JHashNeRF

[Instant-NGP](https://github.com/NVlabs/instant-ngp) recently introduced a Multi-resolution Hash Encoding for neural graphics primitives like [NeRFs](https://www.matthewtancik.com/nerf). The original NVIDIA implementation mainly in C++/CUDA, based on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), can train NeRFs upto 100x faster!

This project is a **pure Jittor** implementation of [Instant-NGP](https://github.com/NVlabs/instant-ngp), built with the purpose of enabling AI Researchers to play around and innovate further upon this method.

This project is built on top of the super-useful [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch)、[HashNeRF-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch)、[jrender](https://github.com/Jittor/jrender/tree/main/jrender/renderer) implementation.

## 简介

本项目包含了第二届计图挑战赛计图-可微渲染新视角生成比赛的代码实现。如上描述，本项目特点是对原始NeRF使用jittor实现了多分辨率的哈希编码，增加了sparse loss和tv loss，添加了一次重要性采样，渲染得图像大致如下。

![Scarr_9.jpg](imgs/Scar_r_9.png)

## 安装

本项目大致需要占用7G显存，在3090上训练时间大约2.5小时。

#### 运行环境

- ubuntu 20.04 LTS
- python >= 3.7
- jittor >= 1.3.0

#### 安装依赖

```
pip install -r requirements.txt
```

#### 数据集下载

```
bash download_competition_data.sh
```

## Train & Refer

```
训练：
bash train.sh
或
python run_nerf.py --config ./configs/Scar.txt
```

```
测试:
python test.py
```

## 致谢

此项目参考了[jrender](https://github.com/Jittor/jrender/tree/main/jrender/renderer)、[HashNeRF-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch)、[NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch)项目，特此致谢。

```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--222103},
  year={2020}
}
```

```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```

```
@misc{bhalgat2022hashnerfpytorch,
  title={HashNeRF-pytorch},
  author={Yash Bhalgat},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yashbhalgat/HashNeRF-pytorch/}},
  year={2022}
}
```



