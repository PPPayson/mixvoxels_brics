import math
import os
import time
from argparse import Namespace

import torch
from tqdm.auto import tqdm
import pdb

import utils
from opt import config_parser
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from functools import partial

import time
import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from dynamics import Dynamics

from dataLoader import dataset_dict
import sys
from torch.profiler import profile, record_function, ProfilerActivity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast
@torch.no_grad()
def render_path(args):
    dataset = dataset_dict['circle']
    test_dataset = dataset(args.datadir, downsample=1, is_stack=True, n_frames=args.n_frames, frame_start=args.frame_start,
                            scene_box=args.scene_box, near=args.near, far=args.far, split=args.split)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray
    if args.temporal_sampler == 'simple':
        temporal_sampler = TemporalSampler(args.n_frames, args.n_train_frames)
    elif args.temporal_sampler == 'weighted':
        temporal_sampler = TemporalWeightedSampler(args.n_frames, args.n_train_frames, args.temperature_start,
                                                   args.temperature_end, args.n_iters, args.temporal_sampler_replace)
    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    # kwargs.update({'device': device})
    # tensorf = eval(args.model_name)(**kwargs)
    tensorf = eval(args.model_name)(args, kwargs['aabb'], kwargs['gridSize'], device,
                                    density_n_comp=kwargs['density_n_comp'], appearance_n_comp=kwargs['appearance_n_comp'],
                                    app_dim=args.data_dim_color, near_far=kwargs['near_far'],
                                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre,
                                    density_shift=args.density_shift, distance_scale=args.distance_scale,
                                    rayMarch_weight_thres=args.rm_weight_mask_thre,
                                    rayMarch_weight_thres_static=args.rm_weight_mask_thre_static,
                                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe,
                                    featureC=args.featureC, step_ratio=kwargs['step_ratio'], fea2denseAct=args.fea2denseAct,
                                    den_dim=args.data_dim_density, densityMode=args.densityMode, featureD=args.featureD,
                                    rel_pos_pe=args.rel_pos_pe, n_frames=args.n_frames,
                                    amp=args.amp, temporal_variance_threshold=args.temporal_variance_threshold,
                                    n_frame_for_static=args.n_frame_for_static,
                                    dynamic_threshold=args.dynamic_threshold, n_time_embedding=args.n_time_embedding,
                                    static_dynamic_seperate=args.static_dynamic_seperate,
                                    zero_dynamic_sigma=args.zero_dynamic_sigma,
                                    zero_dynamic_sigma_thresh=args.zero_dynamic_sigma_thresh,
                                    sigma_static_thresh=args.sigma_static_thresh,
                                    n_train_frames=args.n_train_frames,
                                    net_layer_add=args.net_layer_add,
                                    density_n_comp_dynamic=args.n_lamb_sigma_dynamic,
                                    app_n_comp_dynamic=args.n_lamb_sh_dynamic,
                                    interpolation=args.interpolation,
                                    dynamic_granularity=args.dynamic_granularity,
                                    point_wise_dynamic_threshold=args.point_wise_dynamic_threshold,
                                    static_point_detach=args.static_point_detach,
                                    dynamic_pool_kernel_size=args.dynamic_pool_kernel_size,
                                    time_head=args.time_head, filter_thresh=args.filter_threshold,
                                    static_featureC=args.static_featureC,
                                    )
    tensorf.load(ckpt)
    logfolder = os.path.dirname(args.ckpt)
    cuda_empty()
    c2ws = test_dataset.poses
    target_folder = f'imgs_{args.split}_all'
    
    os.makedirs(f'{logfolder}/{target_folder}', exist_ok=True)
    with torch.no_grad():
        with autocast(enabled=bool(args.amp)):
            evaluation_circle(test_dataset, tensorf, args, c2ws, renderer, f'{logfolder}/{target_folder}/',
                            N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,
                            static_branch_only=args.static_branch_only_initial, temporal_sampler=temporal_sampler,
                            remove_foreground=args.remove_foreground, start_idx=args.render_path_start)


if __name__ == '__main__':
    
    start_time = time.time()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)
    render_path(args)
    print("--- running time %s seconds ---" % (time.time() - start_time))