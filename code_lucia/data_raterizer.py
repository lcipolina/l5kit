# code from here https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/177637

#Imports
#region
#from google.colab import files
from logging import DEBUG
import torch
import gc, os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import torch
from multiprocessing import Pool
import random
import bz2
import pickle
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from torchvision.models.resnet import resnet18
from tqdm import tqdm
from typing import Dict
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,ExponentialLR
from l5kit.evaluation import write_pred_csv
from l5kit.data import LocalDataManager,filter_agents_by_labels,get_combined_scenes
from l5kit.data import ChunkedDataset
from l5kit.dataset import EgoDataset,AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace,rmse,average_displacement_error_mean
# from unofficial_fix_lyft_rester_size import ChunkedDataset,EgoDataset,AgentDataset,build_rasterizer
from pathlib import Path
from tempfile import gettempdir
import pandas as pd
#endregion

#GLOBALS (i.e environment variables)
#Change according to machine computation power
CPUnum = 40

#Data paths
L5KIT_DATA_FOLDER  = '/home/alpha/Desktop/Lucia/l5kit/lyft-motion-prediction-autonomous-vehicles/'  #needed for the call to 'localdata_manager.py'
train_data_path    = '/home/alpha/Desktop/Lucia/l5kit/lyft-motion-prediction-autonomous-vehicles/scenes/train.zarr'
areal_data_path    = '/home/alpha/Desktop/Lucia/l5kit/lyft-motion-prediction-autonomous-vehicles/aerial_map/aerial_map.png'
semantic_data_path = '/home/alpha/Desktop/Lucia/l5kit/lyft-motion-prediction-autonomous-vehicles/semantic_map/semantic_map.pb'
validate_data_path  = '/home/alpha/Desktop/Lucia/l5kit/lyft-motion-prediction-autonomous-vehicles/scenes/validate.zarr'
sample_data_path    = '/home/alpha/Desktop/Lucia/l5kit/lyft-motion-prediction-autonomous-vehicles/scenes/sample.zarr'
results_path ='/home/alpha/Desktop/Lucia/l5kit/results'

#Config Dict
#region
cfg = {
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },

    'raster_params': {
        'raster_size': [350, 350],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': areal_data_path, #LUCIA
        'semantic_map_key': semantic_data_path, #LUCIA
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },

    'train_data_loader': {
        #'key': 'scenes/train.zarr',
        'key': train_data_path,  #LUCIA
        'batch_size': 128,
        'shuffle': True,
        'num_workers': CPUnum # 4  LUCIA
    },

    'val_data_loader': {
        'key': validate_data_path, #LUCIA
        'batch_size': 32,
        'shuffle': False,
        'num_workers': CPUnum #8  LUCIA
    },

    'sample_data_loader': {
        'key': sample_data_path, #UCIA
        'batch_size': 16,
        'shuffle': False,
        'num_workers': CPUnum #0   #LUCIA
    },

    'train_params': {
        'max_num_steps': 438 if DEBUG else 200000,
        'checkpoint_every_n_steps': 10000,

        # 'eval_every_n_steps': -1
    }
}
#endregion

def save_sample(i):
    idx = random.randint(0, len(dataset))

    # 300px, 0.5 raster size, 5 historical frames
    obj_save(dataset[idx], f'sample_{i}', results_path)

def obj_save(obj, name, dir_cache):
    with bz2.BZ2File(f'{dir_cache}/{name}.pbz', 'wb') as f:
        pickle.dump(obj, f)

dm = LocalDataManager(L5KIT_DATA_FOLDER)  #LUCIA - constructor needs to know where to find the data
dataset_path = dm.require(cfg["train_data_loader"]["key"])
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()

rast = build_rasterizer(cfg, dm)
dataset = AgentDataset(cfg, zarr_dataset, rast)

with Pool(processes=4) as p:
    max_ = 32000
    with tqdm(total=max_) as pbar:
        for i, _ in enumerate(p.imap_unordered(save_sample, range(0, max_))):
            pbar.update()

class LyftImageDataset(Dataset):

    def __init__(self, data_folder):
        super().__init__()
        self.data_folder = data_folder
        self.files = []

        for filename in os.listdir(self.data_folder):
            if filename.endswith(".pbz"):
                self.files.append(filename)

        print(len(self.files))
        print(self.files[0])

    def __getitem__(self, index: int):
        return self.obj_load(self.files[index])

    def obj_load(self, name):
        with bz2.BZ2File(f'{self.data_folder}/{name}', 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.files)

train_cfg = cfg["train_data_loader"]
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = LyftImageDataset(results_path)
train_dataloader = DataLoader(train_dataset,
                              shuffle=train_cfg["shuffle"],
                              batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"])
