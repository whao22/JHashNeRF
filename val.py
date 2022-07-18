import numpy as np
import os,json
import jittor as jt
from tqdm import tqdm
import imageio

from run_nerf import batchify,run_network,render_path,config_parser,render,create_nerf
from nerf_helper.load_blender import load_blender_data
from nerf_helper.utils import to8b,get_embedder,NeRF,NeRFSmall


jt.flags.use_cuda = 1
DEBUG = False


def test_poses(basedir, H, W):
    with open(os.path.join(basedir, 'transforms_test.json'), 'r') as fp:
        meta = json.load(fp)

    poses = []
    for frame in meta['frames']:
        a=np.array(frame['transform_matrix'])
        poses.append(a)
    poses = np.array(poses).astype(np.float32)


    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    return poses, [H,W,focal]


def render_path_val(render_poses, hwf, chunk, render_kwargs, savedir=None, render_factor=0, intrinsic = None, expname=""):
    
    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], intrinsic=intrinsic, **render_kwargs)
        if i==0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            rgb8 = to8b(rgb.numpy())
            filename = os.path.join(savedir, expname + '_r_{:d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            
        del rgb
        del disp
        del acc
        del _


def test():
    # 解析参数
    parser = config_parser()
    args = parser.parse_args()

    # 加载数据
    H,W=800,800
    poses, hwf=test_poses(args.datadir,H,W)
    near = args.near
    far = args.far
    print("hwf", hwf)
    print("near", near)
    print("far", far)

    ## Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    render_poses = jt.array(poses)
    ckpt_path = args.ft_path
    print('Reloading from', ckpt_path)
    ckpt = jt.load(ckpt_path)
    args.bounding_box=ckpt['bbox']

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    with jt.no_grad():
        testsavedir = "./result/"
        os.makedirs(testsavedir, exist_ok=True)
        render_path_val(render_poses, hwf, args.chunk, render_kwargs_test, savedir=testsavedir, render_factor=args.render_factor,expname=args.expname)



if __name__=='__main__':
    test()