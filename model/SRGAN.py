# Package Imports
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import numpy as np
import time
from omegaconf import OmegaConf
import wandb

# local imports
from utils.calculate_metrics import calculate_metrics
from utils.logging_helpers import plot_tensors
from utils.logging_helpers import plot_fusion
from utils.logging_helpers import misr_plot
from utils.normalise_s2 import normalise_s2
from utils.dataloader_utils import histogram as histogram_match


#############################################################################################################
# Build PL MODEL


class SRGAN_model(pl.LightningModule):

    def __init__(self, config_file_path="config.yaml"):
        super(SRGAN_model, self).__init__()

        # get config file
        self.config = OmegaConf.load(config_file_path)

        """ IMPORT MODELS """
        # if MISR is wanted, instantiate fusion net
        if self.config.SR_type=="MISR":
            from model.fusion import RecursiveNet,RecursiveNet_pl
            self.fusion = RecursiveNet_pl()
                # load pretrained weights
            if self.config.Model.load_fusion_checkpoint:
                self.fusion = RecursiveNet_pl.load_from_checkpoint(self.config.Model.fusion_ckpt_path, strict=False)

        # Generator
        from model.model_blocks import Generator
        self.generator = Generator(large_kernel_size=self.config.Generator.large_kernel_size,
                        small_kernel_size=self.config.Generator.small_kernel_size,
                        n_channels=self.config.Generator.n_channels,
                        n_blocks=self.config.Generator.n_blocks,
                        scaling_factor=self.config.Generator.scaling_factor)
        
        # Discriminator
        from model.model_blocks import Discriminator
        self.discriminator = Discriminator(kernel_size=self.config.Discriminator.kernel_size,
                            n_channels=self.config.Discriminator.n_channels,
                            n_blocks=self.config.Discriminator.n_blocks,
                            fc_size=self.config.Discriminator.fc_size)
        
        # VGG for encoding
        from model.model_blocks import TruncatedVGG19
        self.truncated_vgg19 = TruncatedVGG19(i=self.config.TruncatedVGG.i, j=self.config.TruncatedVGG.j)
        # freeze VGG19
        for param in self.truncated_vgg19.parameters():
            param.requires_grad = False

        # set up Losses
        self.content_loss_criterion = torch.nn.MSELoss()
        self.adversarial_loss_criterion = torch.nn.BCEWithLogitsLoss()


    def forward(self,lr_imgs):
        # if MISR, perform Fusion first
        if self.config.SR_type=="MISR":
            lr_imgs = self.fusion(lr_imgs)
        # perform generative Step
        sr_imgs = self.generator(lr_imgs)
        return(sr_imgs)
    

    @torch.no_grad()
    def predict(self,lr_imgs):
        """
        This function is for the prediction in the Deployment stage, therefore
        the normalization and denormalization needs to happen here.
        Input:
            - unnormalized lLR imgs
        Output:
            - normalized SR images
        Info:
            - This function currently only performs SISR SR
        """

        # move to GPU if possible
        lr_imgs = lr_imgs.to(self.device)
        # normalize images
        lr_imgs = normalise_s2(lr_imgs,stage="norm")
        # preform SR
        with torch.no_grad():
            sr_imgs = self.generator(lr_imgs)
        # histogram match to also encoded LR images
        sr_imgs = histogram_match(lr_imgs,sr_imgs)
        # denormalize images
        sr_imgs = normalise_s2(sr_imgs,stage="denorm")
        # move to CPU
        sr_imgs = sr_imgs.cpu().detach()
        return sr_imgs


    def training_step(self,batch,batch_idx,optimizer_idx):
        # access data
        lr_imgs,hr_imgs = batch

        # generate SR images, log losses immediately
        sr_imgs = self.forward(lr_imgs)
        metrics = calculate_metrics(sr_imgs,hr_imgs,phase="train")
        for key, value in metrics.items():
            self.log(f'{key}', value)
        
        # Discriminator Step
        if optimizer_idx==0:
            # run discriminator and get loss between pred labels and true labels
            hr_discriminated = self.discriminator(hr_imgs)
            sr_discriminated = self.discriminator(sr_imgs)
            adversarial_loss = self.adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))

            # Binary Cross-Entropy loss
            adversarial_loss = self.adversarial_loss_criterion(sr_discriminated,
                                                            torch.zeros_like(sr_discriminated)) + self.adversarial_loss_criterion(hr_discriminated,
                                                                                                                                    torch.ones_like(hr_discriminated))
            # logg Discriminator loss
            self.log("discriminator/adverserial_loss",adversarial_loss)

            # return weighted discriminator loss
            return adversarial_loss

        # Generator Step
        if optimizer_idx==1:
            
            """ 1. Get VGG space loss """
            # encode images
            sr_imgs_in_vgg_space = self.truncated_vgg19(sr_imgs)
            hr_imgs_in_vgg_space = self.truncated_vgg19(hr_imgs).detach()  # detached because they're constant, targets
            # Calculate the Perceptual loss between VGG encoded images to receive content loss
            content_loss = self.content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
            
            """ 2. Get Discriminator Opinion and loss """
            # run discriminator and get loss between pred labels and true labels
            sr_discriminated = self.discriminator(sr_imgs)
            adversarial_loss = self.adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
            
            """ 3. Weight the losses"""
            perceptual_loss = content_loss + self.config.Losses.adv_loss_beta * adversarial_loss

            """ 4. Log Generator Loss """
            self.log("generator/perceptual_loss",perceptual_loss)
            
            # return Generator loss
            return perceptual_loss
        
    @torch.no_grad()
    def validation_step(self,batch,batch_idx):
        
        """ 1. Extract and Predict """
        lr_imgs,hr_imgs = batch
        sr_imgs = self.forward(lr_imgs)

        # if we're in MISR, keep only 1 image to perform visualization and checks. Keep Fused image for inspection
        if self.config.SR_type=="MISR":
            # get fused image
            lr_fused = self.fusion(lr_imgs)
            # keep only first image for visualization purposes
            lr_imgs_vis = lr_imgs[:,0,:,:,:]

        """ 2. Log Generator Metrics """
        # log image metrics
        metrics_hr_img = torch.clone(hr_imgs)  # deep copy to avoid graph problems
        metrics_sr_img = torch.clone(sr_imgs)  
        metrics = calculate_metrics(metrics_sr_img,metrics_hr_img,phase="val")
        del metrics_hr_img, metrics_sr_img # delete copies from GPU
        for key, value in metrics.items():
            self.log(f'{key}', value)

        # only perform image logging for n pics, not all 200
        if batch_idx<self.config.Logging.num_val_images:
            # log Stadard image visualizations  
            plot_lr_img = torch.clone(lr_imgs_vis)   # deep copy to avoid graph problems
            plot_hr_img = torch.clone(hr_imgs)  
            plot_sr_img = torch.clone(sr_imgs)  
            val_img = plot_tensors(plot_lr_img,plot_sr_img,plot_hr_img,title="Val")
            del plot_lr_img, plot_hr_img, plot_sr_img # delete copies from GPU
            self.logger.experiment.log({"Val SR":  wandb.Image(val_img)}) # log val image

            # log MISR specific image visualizations
            if self.config.SR_type=="MISR":
                # create fusion image
                plot_lr_img = torch.clone(lr_fused)
                # plot images
                plot_fusion_img = plot_fusion(torch.clone(lr_imgs_vis),torch.clone(lr_fused),torch.clone(hr_imgs))
                # log fusion image to logger
                self.logger.experiment.log({"Fusion":  wandb.Image(plot_fusion_img)})

                # plot SR plus grid
                misr_img = misr_plot(torch.clone(lr_imgs),torch.clone(sr_imgs) ,torch.clone(hr_imgs))
                self.logger.experiment.log({"MISR_val":  wandb.Image(misr_img)})
           

        """ 3. Log Discriminator metrics """
        # run discriminator and get loss between pred labels and true labels
        hr_discriminated = self.discriminator(hr_imgs)
        sr_discriminated = self.discriminator(sr_imgs)
        adversarial_loss = self.adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))

        # Binary Cross-Entropy loss
        adversarial_loss = self.adversarial_loss_criterion(sr_discriminated,
                                                        torch.zeros_like(sr_discriminated)) + self.adversarial_loss_criterion(hr_discriminated,
                                                                                                                                torch.ones_like(hr_discriminated))
        self.log("validation/DISC_adversarial_loss",adversarial_loss)


    def on_validation_epoch_end(self):
        # ToDo: fix, log test set image
        pass

    def configure_optimizers(self):

        # configure Generator SISR/MISR optimizers
        optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.generator.parameters()),lr=self.config.Optimizers.optim_g_lr)

        # configure Discriminator optimizers
        optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.discriminator.parameters()),lr=self.config.Optimizers.optim_d_lr)

        # configure schedulers
        scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=self.config.Schedulers.factor_g, patience=self.config.Schedulers.patience_g, verbose=self.config.Schedulers.verbose)
        scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=self.config.Schedulers.factor_d, patience=self.config.Schedulers.patience_d, verbose=self.config.Schedulers.verbose)

        # return schedulers and optimizers
        return [
                    [optimizer_d, optimizer_g],
                    [{'scheduler': scheduler_d, 'monitor': self.config.Schedulers.metric, 'reduce_on_plateau': True, 'interval': 'epoch', 'frequency': 1},
                     {'scheduler': scheduler_g, 'monitor': self.config.Schedulers.metric, 'reduce_on_plateau': True, 'interval': 'epoch', 'frequency': 1}],
                ]

if __name__=="__main__":       
    model = SRGAN_model(config_file_path="config.yaml")