import os, sys
import jittor as jt
import numpy as np
import imageio
import matplotlib.pyplot as plt
from jrender_vol.renderPass import render as render
from run_nerf import create_nerf,config_parser
from nerf_helper.load_blender import load_blender_data
import mcubes

jt.flags.use_cuda=0


scene="Scar"
ckpt_num="050999"


if __name__=="__main__":
    # 加载参数
    config_file=f"./configs/{scene}.txt"
    ft_path=f"./logs/{scene}/{ckpt_num}.tar"
    print(f"加载配置文件 {config_file}, 加载模型 {ft_path}")
    parser = config_parser()
    args = parser.parse_args(f'--config {config_file} --ft_path {ft_path}')
    images, poses, render_poses, hwf, i_split, bbox = load_blender_data(args.datadir, args.near,args.far, args.half_res, args.testskip, args.blender_factor)
    args.bounding_box = bbox

    # 创建nerf模型
    _, render_kwargs_test, start, grad_vars, models = create_nerf(args)
    bds_dict = {
        'near' : args.near,
        'far' : args.far,
    }
    render_kwargs_test.update(bds_dict)
    net_fn = render_kwargs_test['network_query_fn']

    # 采样
    N = 256
    t = jt.linspace(-2.5, 2.5, N+1)
    query_pts = jt.stack(jt.meshgrid(t, t, t), -1)
    print(query_pts.shape)
    sh = query_pts.shape
    flat = query_pts.reshape([-1,3])
    
    fn = lambda i0, i1 : net_fn(flat[i0:i1,None,:], viewdirs=jt.zeros_like(flat[i0:i1]), network_fn=render_kwargs_test['network_fine1'])
    chunk = 1024*64
    raw = jt.concat([fn(i, i+chunk).numpy() for i in range(0, flat.shape[0], chunk)], 0)
    raw = jt.reshape(raw, list(sh[:-1]) + [-1])
    sigma = jt.maximum(raw[...,-1], 0.)
    print(sigma.shape)

    # marching cube
    threshold = 10.
    print('fraction occupied', jt.mean(sigma > threshold))
    vertices, triangles = mcubes.marching_cubes(sigma.numpy(), threshold)
    mcubes.export_obj(vertices, triangles, f'{scene}.obj')
    print('done', vertices.shape, triangles.shape)
