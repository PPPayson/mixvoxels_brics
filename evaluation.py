import random
import subprocess
import shlex
from skimage.metrics import structural_similarity as sk_ssim
import numpy as np
import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
from argparse import Namespace
import multiprocessing
import imageio

def evaluation(test_path, gt_path, test_start_frame, gt_start_frame, num_frames):
    print('=================EVAL==================')
    PSNRs, PSNRs_pf, PSNRs_STA = [], [], []
    
    
    comp_frames_path = os.path.join(test_path, "images")
    test_static_rgb = None
    
    for test_f in os.listdir(test_path):
        if test_f.endswith('_static_rgb.png'):
            test_static_rgb = imageio.imread(os.path.join(test_path, test_f))
    frame_count = 0
    test_rgbs, gt_rgbs = [], []
    for test_f in os.listdir(comp_frames_path):
        if test_f.endswith(('.png', '.jpg')):
            frame_count += 1
            if frame_count >= test_start_frame and len(test_rgbs)<num_frames:
                test_rgbs.append(imageio.imread(os.path.join(comp_frames_path, test_f)))
    
    frame_count = 0
    for gt_f in os.listdir(gt_path)
        if gt_f.endswith(('png', 'jpg')):
            frame_count += 1
            if frame_count >= gt_start_frame and len(gt_rgbs)<num_frames:
                gt_rgbs.append(imageio.imread(gt_f))
                
    test_rgbs = np.vstack(test_rgbs)
    gt_rgbs = np.vstack(gt_rgbs)
    #print(test_rgbs.shape)
    #print(gt_rgbs.shape)
    gt_static_rgb = gt_rgbs.mean(dim=2)
    per_frame_loss = ((test_rgbs-gt_rgb) ** 2).mean(dim=0).mean(dim=0).mean(dim=1)
    loss = per_frame_loss.mean()
    loss_static = np.mean((test_static_rgb - gt_static_rgb) ** 2)
    
    PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))
    PSNRs_pf.append((-10.0 * np.log(per_frame_loss.detach().cpu().numpy()) / np.log(10.0)).mean())
    PSNRs_STA.append(-10.0 * np.log(loss_static.item()) / np.log(10.0))
    
    for i_time in range(0, len(test_rgbs), 10):
        ssim = sk_ssim(test_rgb[i_time], gt_rgbs[i_time])
        
    
    
    
    