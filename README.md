# Jittor 可微渲染新视角生成比赛 JHashNeRF

[Instant-NGP](https://github.com/NVlabs/instant-ngp) recently introduced a Multi-resolution Hash Encoding for neural graphics primitives like [NeRFs](https://www.matthewtancik.com/nerf). The original NVIDIA implementation mainly in C++/CUDA, based on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), can train NeRFs upto 100x faster!
This project is a **pure Jittor** implementation of [Instant-NGP](https://github.com/NVlabs/instant-ngp), built with the purpose of enabling AI Researchers to play around and innovate further upon this method.
This project is built on top of the super-useful [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch)、[HashNeRF-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch)、[jrender](https://github.com/Jittor/jrender/tree/main/jrender/renderer) implementation.

## Description
This project contains the code implementation for the second Computational Graphics Challenge - Differentiable Rendering for Novel View Synthesis. As described above, the notable aspects of this project involve the utilization of Jittor to implement multi-resolution hashing encoding on the original NeRF (Neural Radiance Fields) model. It includes the addition of sparse loss and TV loss, as well as incorporating one-time importance sampling. The rendered images roughly resemble the following.

<div align=center>
  <img src="imgs/Scar_r_9.png" height="300"/>
<!-- ![Scarr_9.jpg](imgs/Scar_r_9.png) -->
</div>

## Uasge

### Requirements
- ubuntu 20.04 LTS
- python >= 3.7
- jittor >= 1.3.0

Please run the following script to create the environment
```
pip install -r requirements.txt
```

### Dataset

```
bash download_competition_data.sh
```

### Train

Run the following script to train 
```
bash train.sh
# or
python run_nerf.py --config ./configs/Scar.txt
```

## Eval
```
python val.py --config ./configs/$scene.txt --ft_path=lohs/$scene.tar
```

## Acknowledgement

The project bases on [jrender](https://github.com/Jittor/jrender/tree/main/jrender/renderer),  [HashNeRF-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch), [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch). Thanks to all of them.

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



