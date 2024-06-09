# Imports
import os,torch
from omegaconf import OmegaConf
from model.SRGAN import SRGAN_model
import opensr_utils

# set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Load Model
config = OmegaConf.load("configs/config_3band.yaml")
model = SRGAN_model(config).cuda()

# Set up sr utils
from opensr_utils import windowed_SR_and_saving
file_path = "/data1/simon/datasets/val_s2_tiles/S2A_MSIL2A_20240214T031831_N0510_R118_T48PXT_20240214T063445.SAFE/"
sr_obj = windowed_SR_and_saving(file_path, window_size=(75, 75), factor=4, keep_lr_stack=True,mode="SR")
sr_obj.start_super_resolution(band_selection="10m",model=model,forward_call="forward",overlap=20, eliminate_border_px=0)
 


