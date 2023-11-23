# Package Imports
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import numpy as np
import matplotlib.pyplot as plt
import time
from omegaconf import OmegaConf
import wandb
from torch.utils.data import DataLoader
import datetime
import os
from multiprocessing import freeze_support

# local imports
from utils.calculate_metrics import calculate_metrics
from utils.logging_helpers import plot_tensors
from utils.datasets import cv_dataset
from utils.datasets import create_pl_datamodule
from model.SRGAN import SRGAN_model


# __main__ 
if __name__ == '__main__':
    # required for Multprocessing on Windows
    freeze_support()

    # General
    torch.set_float32_matmul_precision('medium')
    # load config
    config = OmegaConf.load("config.yaml")

    #############################################################################################################
    " LOAD MODEL "
    #############################################################################################################

    if config.Model.load_checkpoint:
        model = SRGAN_model.load_from_checkpoint(config.Model.ckpt_path, strict=False)
    else:
        model = SRGAN_model()

    #############################################################################################################
    " GET DATA "
    #############################################################################################################
    # create dataloaders via dataset_selector -> config -> class selection -> convert to pl_module
    from utils.datasets import dataset_selector
    pl_datamodule = dataset_selector(config)

    #############################################################################################################
    """ Configure Trainer """
    #############################################################################################################
    # set up logging
    from pytorch_lightning.loggers import WandbLogger
    wandb_project = "2023_SRGAN"
    wandb_logger = WandbLogger(project=wandb_project,entity="simon-donike")

    from pytorch_lightning import loggers as pl_loggers
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.normpath("logs/"))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.normpath('logs/tmp'))

    from pytorch_lightning.callbacks import ModelCheckpoint
    dir_save_checkpoints = os.path.join(tb_logger.save_dir,wandb_project,
                                                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print("Experiment Path:",dir_save_checkpoints)

    checkpoint_callback = ModelCheckpoint(dirpath=dir_save_checkpoints,
                                            monitor='val/L1',
                                        mode='min',
                                        save_last=True,
                                        save_top_k=2)

    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # callback to set up early stopping
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    early_stop_callback = EarlyStopping(monitor="val/L1", min_delta=0.00, patience=250, verbose=True,
                                    mode="min",check_finite=True) # patience in epochs

    #############################################################################################################
    """ Start Training """
    #############################################################################################################
    
    trainer = Trainer(accelerator='cuda',
                    devices=[0],
                    check_val_every_n_epoch=1,
                    val_check_interval=0.5,
                    limit_val_batches=250,
                    max_epochs=99999,
                    logger=[ 
                                wandb_logger,
                            ],
                    callbacks=[ checkpoint_callback,
                                early_stop_callback,
                                lr_monitor
                                ])


    trainer.fit(model, datamodule=pl_datamodule)
    wandb.finish()
    writer.close()

