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
from utils.normalise_s2 import normalise_s2



#############################################################################################################
# Build PL MODEL


class SRGAN_model(pl.LightningModule):

    def __init__(self, config_file="config.yaml"):
        super(SRGAN_model, self).__init__()

        # get config file
        self.config = OmegaConf.load("config.yaml")

        """ IMPORT MODELS """
        # if MISR is wanted, instantiate fusion net
        if self.config.SR_type=="MISR":
            from model.fusion import RecursiveNet
            self.fusion = RecursiveNet()

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

    def training_step(self,batch,batch_idx,optimizer_idx):
        # access data
        lr_imgs,hr_imgs = batch

        # perform Fusion warmup step if necessary
        """
        if self.check_for_warmup_step(optimizer_idx): # check wether all conditions for warmup are met
            warmup_loss = self.warmup_step(lr_imgs,optimizer_idx) # get loss  
            return(warmup_loss)
        """

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
            # keep only first image
            lr_imgs = lr_imgs[:,0,:,:,:]

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
            plot_lr_img = torch.clone(lr_imgs)   # deep copy to avoid graph problems
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
                plot_fusion_img = plot_fusion(torch.clone(lr_imgs),torch.clone(lr_fused),torch.clone(hr_imgs))
                # log fusion image to logger
                self.logger.experiment.log({"Fusion":  wandb.Image(plot_fusion_img)})
           

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
    
    
    """ MISR Fusion Warmup """
    def check_for_warmup_step(self,optimizer_idx):
        """
        Checks whether the training is currently in the warm-up phase for a Multi-image Super-Resolution (MISR) model.

        This function determines if the model is in the warm-up phase based on the current epoch, the optimizer being used, 
        and specific configurations set for MISR and warm-up. During the warm-up phase, it freezes the parameters of both the 
        generator and discriminator to prevent their update. Once the warm-up phase is over, it unfreezes these parameters to 
        allow normal training.

        Args:
            optimizer_idx (int): The index of the current optimizer. This is used to check if the function is being called for the 
                                correct optimizer (typically, 0 for the generator).

        Returns:
            bool: A boolean value indicating whether the model is in the warm-up phase. Returns `True` if in warm-up phase, 
                otherwise `False`.

        Note:
            - The function relies on the `self.config` attribute of the class to access configuration settings for MISR and warm-up.
            - It specifically checks if the `SR_type` is set to 'MISR' and if warm-up is enabled in the configuration.
            - The number of epochs for the warm-up phase is also retrieved from the configuration (`self.config.Warmup.fusion.epochs`).
            - During the warm-up phase (when this function returns `True`), the gradients for the generator and discriminator 
            are disabled by setting `requires_grad` to `False`. Post warm-up, `requires_grad` is set to `True` to resume normal training.
        """
        MISR_and_warmup = self.config.SR_type=="MISR" and self.config.Warmup.fusion.enable
        epoch0_and_optimizer0 = self.current_epoch<self.config.Warmup.fusion.epochs and optimizer_idx==0
        perform_warmup = MISR_and_warmup and epoch0_and_optimizer0

        # freeze parts of the model depending on wether we're in warmup or not
        if perform_warmup:
            for param in self.generator.parameters():
                param.requires_grad = False
            for param in self.discriminator.parameters():
                param.requires_grad = False
        if not perform_warmup:
            for param in self.generator.parameters():
                param.requires_grad = True
            for param in self.discriminator.parameters():
                param.requires_grad = True

        return perform_warmup
    
    def warmup_step(self,lr_imgs,optimizer_idx):
        """
        Performs a warm-up step by fusing low-resolution (LR) images and calculating the loss between the fused image and the 
        mean of the input LR images.

        This function is part of the warm-up phase in training, where the goal is to align the fused output of the network with 
        the mean of the input images. This helps in stabilizing the training process in the initial stages.

        Args:
            lr_imgs (torch.Tensor): A tensor containing a batch of low-resolution images. The tensor is expected to have a shape 
                                    compatible with the `fusion` module of the model.

        Returns:
            torch.Tensor: A scalar tensor representing the mean squared error loss between the fused LR image and the mean of 
                        the input LR images.

        Note:
            - The function first applies the `fusion` module to the input LR images to get the fused LR image.
            - It then calculates the mean of the input LR images along the 'views' dimension.
            - The mean squared error (MSE) loss is computed between the fused image and the mean image.
            - This loss is used during the warm-up phase of training to guide the fusion process.
        """

        if optimizer_idx==1: # if we're in the right Optim, return Loss
            # perform fusion
            lr_imgs_fused = self.fusion(lr_imgs)
            # create groud truth (mean of input images)
            lr_imgs_mean = torch.mean(lr_imgs,dim=1,keepdim=True).squeeze()
            # calculate loss between fused image and mean of LR images
            warmup_loss = torch.nn.MSELoss()(lr_imgs_fused,lr_imgs_mean)
            #  log and return loss
            self.log("warmup/fusion_warmup",warmup_loss)
            return(warmup_loss)
        else:
            return None
        



if __name__=="__main__":       
    m = SRGAN_model()