import jittor as jt
from jittor import nn
import numpy as np
from nerf_helper.hash_encoder import HashEmbedder,SHEncoder
from jrender_vol.camera.pinhole import pinhole_get_rays

# Misc
img2mse = lambda x, y : jt.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * jt.log(x) / jt.log(jt.array(np.array([10.])))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**jt.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = jt.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return jt.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, args, i=0):
    if i == -1:
        return nn.Identity(), 3
    elif i==0:
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [jt.sin, jt.cos],
        }
        
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        out_dim = embedder_obj.out_dim
    elif i==1:
        embed = HashEmbedder(bounding_box=args.bounding_box, \
                            log2_hashmap_size=args.log2_hashmap_size, \
                            finest_resolution=args.finest_res)
        out_dim = embed.out_dim
    elif i==2:
        embed = SHEncoder()
        out_dim = embed.out_dim
    return embed, out_dim

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def execute(self, x):
        input_pts, input_views = jt.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = jt.nn.relu(h)
            if i in self.skips:
                h = jt.concat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = jt.concat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = jt.nn.relu(h)

            rgb = self.rgb_linear(h)
            outputs = jt.concat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = jt.array(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = jt.array(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = jt.array(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = jt.array(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = jt.array(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = jt.array(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = jt.array(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = jt.array(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = jt.array(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = jt.array(np.transpose(weights[idx_alpha_linear+1]))



# # Small NeRF for Hash embeddings
# class NeRFSmall(nn.Module):
#     def __init__(self,
#                  num_layers=3,
#                  hidden_dim=64,
#                  geo_feat_dim=15,
#                  num_layers_color=4,
#                  hidden_dim_color=64,
#                  input_ch=3, input_ch_views=3,
#                  ):
#         super(NeRFSmall, self).__init__()

#         self.input_ch = input_ch
#         self.input_ch_views = input_ch_views

#         # sigma network
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.geo_feat_dim = geo_feat_dim

#         sigma_net = []
#         for l in range(num_layers):
#             if l == 0:
#                 in_dim = self.input_ch
#             else:
#                 in_dim = hidden_dim
            
#             if l == num_layers - 1:
#                 out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
#             else:
#                 out_dim = hidden_dim
            
#             sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

#         self.sigma_net = nn.ModuleList(sigma_net)

#         # color network
#         self.num_layers_color = num_layers_color        
#         self.hidden_dim_color = hidden_dim_color
        
#         color_net =  []
#         for l in range(num_layers_color):
#             if l == 0:
#                 in_dim = self.input_ch_views + self.geo_feat_dim
#             else:
#                 in_dim = hidden_dim
            
#             if l == num_layers_color - 1:
#                 out_dim = 3 # 3 rgb
#             else:
#                 out_dim = hidden_dim
            
#             color_net.append(nn.Linear(in_dim, out_dim, bias=False))

#         self.color_net = nn.ModuleList(color_net)
    
#     def execute(self, x):
#         input_pts, input_views = jt.split(x, [self.input_ch, self.input_ch_views], dim=-1)

#         # sigma
#         h = input_pts
#         for l in range(self.num_layers):
#             h = self.sigma_net[l](h)
#             if l != self.num_layers - 1:
#                 h = nn.relu(h)

#         # sigma, geo_feat = h[..., 0], h[..., 1:]
#         sigma, geo_feat = h[..., 0], h[..., 1:]
        
        
#         # color
#         h = jt.concat([input_views, geo_feat], dim=-1)
#         for l in range(self.num_layers_color):
#             h = self.color_net[l](h)
#             if l != self.num_layers_color - 1:
#                 h = nn.relu(h)
            
#         # color = torch.sigmoid(h)
#         color = h
#         outputs = jt.concat([color, sigma.unsqueeze(dim=-1)], -1)

#         return outputs

#Small NeRF for Hash embeddings
class NeRFSmall(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,
                 ):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)
    
    def execute(self, x):
        input_pts, input_views = jt.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # sigma
        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = nn.relu(h)

        sigma, geo_feat = h[..., 0], h[..., 1:]
        
        
        # color
        h = jt.concat([input_views, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = nn.relu(h)
            
        # color = torch.sigmoid(h)
        color = h
        outputs = jt.concat([color, sigma.unsqueeze(dim=-1)], -1)

        return outputs



# # Ray helpers
# def get_rays(H, W, focal, c2w, intrinsic = None):
#     i, j = jt.meshgrid(jt.linspace(0, W-1, W), jt.linspace(0, H-1, H))
#     i = i.t()
#     j = j.t()
#     if intrinsic is None:
#         dirs = jt.stack([(i-W*.5)/focal, (j-H*.5)/focal, jt.ones_like(i)], -1).unsqueeze(-2)
#     else:
#         i+=0.5
#         j+=0.5
#         dirs = jt.stack([i, j, jt.ones_like(i)], -1).unsqueeze(-2)
#         dirs = jt.sum(dirs * intrinsic[:3,:3], -1).unsqueeze(-2)
#     # Rotate ray directions from camera frame to the world frame
#     rays_d = jt.sum(dirs * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = c2w[:3,-1].expand(rays_d.shape)
#     return rays_o, rays_d

def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t.unsqueeze(-1) * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = jt.stack([o0,o1,o2], -1)
    rays_d = jt.stack([d0,d1,d2], -1)

    return rays_o, rays_d



def get_bbox3d_for_blenderobj(camera_transforms, H, W, near=2.0, far=6.0):
    camera_angle_x = float(camera_transforms['camera_angle_x'])
    focal = 0.5*W/np.tan(0.5 * camera_angle_x)

    min_bound = jt.Var([100, 100, 100])
    max_bound = jt.Var([-100, -100, -100])

    points = []

    for frame in camera_transforms["frames"]:
        c2w = jt.Var(frame["transform_matrix"])
        rays_o, rays_d = pinhole_get_rays(H,W,focal, c2w)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)

        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (jt.Var(min_bound)-jt.Var([1.0,1.0,1.0]), jt.Var(max_bound)+jt.Var([1.0,1.0,1.0]))
