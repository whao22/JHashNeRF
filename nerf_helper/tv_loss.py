# Author: Yash Bhalgat

from math import exp, log, floor
import jittor as jt
import jittor.nn as nn
import pdb

from .hash_encoder import hash


def total_variation_loss(embeddings, min_resolution, max_resolution, level, log2_hashmap_size, n_levels=16):
    # Get resolution
    b = jt.exp((jt.log(max_resolution)-jt.log(min_resolution))/(n_levels-1))
    resolution = jt.floor(min_resolution * b**level)

    # Cube size to apply TV loss
    min_cube_size = min_resolution - 1
    max_cube_size = 50 # can be tuned
    if min_cube_size > max_cube_size:
        print("ALERT! min cuboid size greater than max!")
        pdb.set_trace()
    cube_size = jt.floor(jt.clamp(resolution/10.0, min_cube_size, max_cube_size)).int32()

    # Sample cuboid
    min_vertex = jt.randint(0, resolution-cube_size, (3,))
    idx = min_vertex + jt.stack([jt.arange(int(cube_size+1)) for _ in range(3)], dim=-1)
    cube_indices = jt.stack(jt.meshgrid(idx[:,0], idx[:,1], idx[:,2]), dim=-1)

    hashed_indices = hash(cube_indices, log2_hashmap_size)
    cube_embeddings = embeddings(hashed_indices)
    #hashed_idx_offset_x = hash(idx+torch.tensor([1,0,0]), log2_hashmap_size)
    #hashed_idx_offset_y = hash(idx+torch.tensor([0,1,0]), log2_hashmap_size)
    #hashed_idx_offset_z = hash(idx+torch.tensor([0,0,1]), log2_hashmap_size)

    # Compute loss
    #tv_x = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_x), 2).sum()
    #tv_y = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_y), 2).sum()
    #tv_z = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_z), 2).sum()
    tv_x = jt.pow(cube_embeddings[1:,:,:,:]-cube_embeddings[:-1,:,:,:], 2).sum()
    tv_y = jt.pow(cube_embeddings[:,1:,:,:]-cube_embeddings[:,:-1,:,:], 2).sum()
    tv_z = jt.pow(cube_embeddings[:,:,1:,:]-cube_embeddings[:,:,:-1,:], 2).sum()

    return (tv_x + tv_y + tv_z)/cube_size

def sigma_sparsity_loss(sigmas):
    # Using Cauchy Sparsity loss on sigma values
    return jt.log(1.0 + 2*sigmas**2).sum(dim=-1)

def huber_loss(x,tar,delta):
    rel=jt.abs(x-tar)
    sqr = 0.5/delta*rel*rel
    return jt.ternary((rel > delta), rel-0.5*delta, sqr).mean()