import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import math
import os
import cv2
import json
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from utils import get_ray_weight, SimpleSampler
import time
from .ray_utils import *


class RenderCircleDataset(Dataset):
    def __init__(self, datadir, split='circle', downsample=1, is_stack=False,
                 tmp_path='memory', scene_box=[-3.0, -3.0, -3.0],
                 near=0.1, far=15.0, frame_start=0, n_frames=150):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.downsample = downsample
        self.frame_start = frame_start
        self.n_frames = n_frames
        self.cam_nums = []
        self.root_dir = datadir
        print("Root:", self.root_dir)
        self.split = split
        print("Split:", self.split)
        self.tmp_path = tmp_path
        print("Tmp path:", tmp_path)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        print("blender2opencv:", self.blender2opencv)
        self.near_far = [near, far]
        print("Near-far:", self.near_far)
        self.scene_bbox = torch.tensor([scene_box, list(map(lambda x: -x, scene_box))])
        print(self.scene_bbox)
        self.read_meta()
        #self.define_proj_mat()
        self.white_bg = True
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        
    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)
        
        w, h = int(self.meta['frames'][0]['w']/self.downsample), int(self.meta['frames'][0]['h']/self.downsample)
        self.img_wh = [w,h]
        assert(self.downsample == 1.0)
        
        self.directions = []
        self.intrinsics = []
        self.focal_x = []
        self.focal_y = []
        self.cx = []
        self.cy = []
        self.meta['frames'] = self.meta['frames'][self.frame_start:self.frame_start+self.n_frames]
        for i in range(0, self.n_frames):
            self.focal_x.append(self.meta['frames'][i]['fl_x'])
            self.focal_y.append(self.meta['frames'][i]['fl_y'])
            self.cx.append(self.meta['frames'][i]['cx'])
            self.cy.append(self.meta['frames'][i]['cy'])
            self.directions.append(get_ray_directions(h, w, [self.focal_x[i], self.focal_y[i]], center=[self.cx[i], self.cy[i]]))
            self.intrinsics.append(torch.tensor([[self.focal_x[i],0,self.cx[i]],[0,self.focal_y[i],self.cy[i]],[0,0,1]]).float())
        
        self.all_rays = []
        self.all_rgbs = []
        self.all_stds_without_diffusion = []
        self.all_rays_weight = []
        self.all_stds = []
        self.poses = []
        
        idxs = list(range(0, self.n_frames))
        #idxs = list(range(0, len(self.meta['frames'])))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):
            # get c2w 
            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]
            self.cam_nums.append(frame['file_path'])
        
        self.poses = torch.stack(self.poses)
        

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}

        return sample


    
