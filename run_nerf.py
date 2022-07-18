import os, sys
import numpy as np
import imageio
import json
import random
import time
import jittor as jt
from jittor import nn
from tqdm import tqdm, trange
import datetime
import matplotlib.pyplot as plt

from nerf_helper.utils import *
from nerf_helper.load_llff import load_llff_data
from nerf_helper.load_deepvoxels import load_dv_data
from nerf_helper.load_blender import load_blender_data
from nerf_helper.tv_loss import total_variation_loss,huber_loss

from tensorboardX import SummaryWriter
from jrender_vol.renderPass import render as render
from jrender_vol.camera import *

jt.flags.use_cuda = 1
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        arr = []
        for i in range(0, inputs.shape[0], chunk):
            arr.append(fn(inputs[i:i+chunk]))
        return jt.concat(arr, 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = jt.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = jt.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = jt.concat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = jt.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, intrinsic = None, expname=""):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], intrinsic=intrinsic, **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i==0:
            print(rgb.shape, disp.shape)


        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, expname + '_r_{:d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
        del rgb
        del disp
        del acc
        del _


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args, args.i_embed)
    embedding_params=[]
    if args.i_embed==1:
            # hashed embedding table
        embedding_params+= list(embed_fn.parameters())

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args, args.i_embed_views)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    if args.i_embed!=1:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    else:
        model = NeRFSmall(num_layers=2,
                        hidden_dim=args.dim_hidden,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=args.dim_hidden,
                        input_ch=input_ch, input_ch_views=input_ch_views)

    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        if args.i_embed==1:
            model_fine = NeRFSmall(num_layers=2,
                        hidden_dim=args.dim_hidden,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=args.dim_hidden,
                        input_ch=input_ch, input_ch_views=input_ch_views)

            model_fine1 = NeRFSmall(num_layers=2,
                        hidden_dim=args.dim_hidden,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=args.dim_hidden,
                        input_ch=input_ch, input_ch_views=input_ch_views)
            
        else:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                            input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
            model_fine1 = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                            input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        
        grad_vars += list(model_fine.parameters())
        grad_vars += list(model_fine1.parameters())


    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    if args.i_embed==1:
        optimizer = jt.optim.Adam([
                            {'params': grad_vars, 'weight_decay': 1e-6},
                            {'params': embedding_params, 'eps': 1e-15}
                        ], lr=args.lrate, betas=(0.9, 0.99))
    else:
        optimizer = jt.optim.Adam(params=grad_vars+embedding_params, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    ##########################
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = jt.load(ckpt_path)
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            model_fine1.load_state_dict(ckpt['network_fine1_state_dict'])
        if args.i_embed==1:
            embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])


    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'network_fine1' : model_fine1,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'embed_fn': embed_fn,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer




def config_parser():
    gpu = "gpu"+os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/'+gpu+"/", 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=1e-2, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=1, 
                        help='set 0 for default positional encoding, -1 for none, 1 for hash encoder')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--faketestskip", type=int, default=1, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--near", type=float, default=2., 
                        help='set near distance')
    parser.add_argument("--far", type=float, default=6., 
                        help='set far distance')
    parser.add_argument("--do_intrinsic", action='store_true', 
                        help='use intrinsic matrix')
    parser.add_argument("--blender_factor", type=int, default=1, 
                        help='downsample factor for blender images')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=5000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=5000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10000,
                        help='frequency of testset saving')
    parser.add_argument("--i_tottest", type=int, default=400000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=500000, # 设置了一个很大的数，用以不生成视频
                        help='frequency of render_poses video saving')

    ###########
    parser.add_argument("--i_embed_views", type=int, default=2, 
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--finest_res",   type=int, default=2048, 
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size",   type=int, default=19, 
                        help='log2 of hashmap size')
    parser.add_argument("--sparse_loss_weight", type=float, default=1e-10,
                        help='learning rate')
    parser.add_argument("--tv_loss_weight", type=float, default=1e-6,
                        help='learning rate')
    parser.add_argument("--delta", type=float, default=0.1,
                        help='huber loss delta')
    parser.add_argument("--bb_scale", type=float, default=0.33,
                        help='boundingbox scale')
    parser.add_argument("--iters", type=int, default=50000,
                        help='num of iter')
    
    parser.add_argument("--dim_hidden", type=int, default=64, help="隐层维度, 其实64就够了, 因为在另一台机器上改了128一次看显存,忘记恢复了,所以多添了当前这个参数")
    

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    intrinsic = None
    if args.dataset_type == 'llff':
        pass
    elif args.dataset_type == 'blender':
        testskip = args.testskip
        faketestskip = args.faketestskip
        if jt.mpi and jt.mpi.local_rank()!=0:
            testskip = faketestskip
            faketestskip = 1
        if args.do_intrinsic:
            images, poses, intrinsic, render_poses, hwf, i_split ,bbox= load_blender_data(args.datadir,args.near,args.far, args.half_res, args.testskip, args.blender_factor, True)
        else:
            images, poses, render_poses, hwf, i_split, bbox = load_blender_data(args.datadir, args.near,args.far, args.half_res, args.testskip, args.blender_factor)
        args.bounding_box = (bbox[0] * args.bb_scale,bbox[1] * args.bb_scale)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        i_test_tot = i_test
        i_test = i_test[::args.faketestskip]

        near = args.near
        far = args.far
        print(args.do_intrinsic)
        print("hwf", hwf)
        print("near", near)
        print("far", far)

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':
        pass
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = jt.array(render_poses)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with jt.no_grad():
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test, savedir=testsavedir, render_factor=args.render_factor,expname=expname)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    accumulation_steps = 1
    N_rand = args.N_rand//accumulation_steps
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    images = jt.array(images.astype(np.float32))
    poses = jt.array(poses)
    if use_batching:
        rays_rgb = jt.array(rays_rgb)


    
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    if not jt.mpi or jt.mpi.local_rank()==0:
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                        .replace(":", "")\
                                        .replace(" ", "_")
        gpu_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        log_dir = os.path.join("./logs", "summaries", "log_" + date +"_gpu" + gpu_idx)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    start = start + 1
    N_iters = start+args.iters
    for i in trange(start, N_iters):
        # jt.display_memory_info()
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = jt.transpose(batch, (1, 0, 2))
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = jt.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            np.random.seed(i)
            img_i = np.random.choice(i_train)
            target = images[img_i]#.squeeze(0)
            pose = poses[img_i, :3,:4]#.squeeze(0)
            if N_rand is not None:
                rays_o, rays_d = pinhole_get_rays(H, W, focal, pose, intrinsic)# (H, W, 3), (H, W, 3)
                
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = jt.stack(
                        jt.meshgrid(
                            jt.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            jt.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = jt.stack(jt.meshgrid(jt.linspace(0, H-1, H), jt.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = jt.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].int()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = jt.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        # img_loss = img2mse(rgb, target_s)
        img_loss = huber_loss(rgb,target_s,args.delta)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img2mse(rgb, target_s))

        if 'rgb0' in extras:
            # img_loss0 = img2mse(extras['rgb0'], target_s)
            img_loss0 = huber_loss(extras['rgb0'], target_s, args.delta)
            loss = loss + img_loss0
            # psnr0 = mse2psnr(img2mse(rgb, target_s))
        
        if 'rgb1' in extras:
            # img_loss0 = img2mse(extras['rgb0'], target_s)
            img_loss1 = huber_loss(extras['rgb1'], target_s, args.delta)
            loss = loss + img_loss1
            # psnr1 = mse2psnr(img2mse(rgb, target_s))
        
        sparsity_loss = args.sparse_loss_weight*(extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum())
        loss = loss + sparsity_loss

        # add Total Variation loss
        if args.i_embed==1 and i<1000:
            n_levels = render_kwargs_train["embed_fn"].n_levels
            min_res = render_kwargs_train["embed_fn"].base_resolution
            max_res = render_kwargs_train["embed_fn"].finest_resolution
            log2_hashmap_size = render_kwargs_train["embed_fn"].log2_hashmap_size
            TV_loss = sum(total_variation_loss(render_kwargs_train["embed_fn"].embeddings[i], \
                                              min_res, max_res, \
                                              i, log2_hashmap_size, \
                                              n_levels=n_levels) for i in range(n_levels))
            loss = loss + args.tv_loss_weight * TV_loss


        optimizer.backward(loss / accumulation_steps)
        if i % accumulation_steps == 0:
            optimizer.step()
        
        ###   update learning rate   ###
        decay_rate = 0.8
        decay_steps = args.lrate_decay * accumulation_steps * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0

        # Rest is logging
        if (i+1)%args.i_weights==0 and (not jt.mpi or jt.mpi.local_rank()==0):
            print(i)
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.i_embed==1:
                jt.save({
                    'global_step': global_step,
                    'bbox': args.bounding_box,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'network_fine1_state_dict': render_kwargs_train['network_fine1'].state_dict(),
                    'embed_fn_state_dict': render_kwargs_train['embed_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                jt.save({
                    'global_step': global_step,
                    'bbox': args.bounding_box,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'network_fine1_state_dict': render_kwargs_train['network_fine1'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with jt.no_grad():
                rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test, intrinsic = intrinsic,expname=expname)
            if not jt.mpi or jt.mpi.local_rank()==0:
                print('Done, saving', rgbs.shape, disps.shape)
                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
                print('movie base ', moviebase)
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            if i%args.i_img==0:
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with jt.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, intrinsic=intrinsic,
                                                        **render_kwargs_test)
                psnr = mse2psnr(img2mse(rgb, target))
                rgb = rgb.numpy()
                disp = disp.numpy()
                acc = acc.numpy()

                if not jt.mpi or jt.mpi.local_rank()==0:
                    writer.add_image('val/rgb', to8b(rgb), global_step, dataformats="HWC")
                    writer.add_image('val/target', target.numpy(), global_step, dataformats="HWC")
                    writer.add_scalar('val/psnr', psnr.item(), global_step)
                
            jt.clean_graph()
            jt.sync_all()
            jt.gc()
        

            if i%args.i_testset==0 and i > 0:
                si_test = i_test_tot if i%args.i_tottest==0 else i_test
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', poses[si_test].shape)
                with jt.no_grad():
                    rgbs, disps = render_path(jt.array(poses[si_test]), hwf, args.chunk, render_kwargs_test, savedir=testsavedir, intrinsic = intrinsic, expname = expname)
                jt.gc()
        global_step += 1


if __name__=='__main__':
    train()