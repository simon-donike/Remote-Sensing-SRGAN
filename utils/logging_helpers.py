import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
import torchvision.transforms as transforms
from utils.sen2_stretch import sen2_stretch
from utils.normalise_s2 import normalise_s2
from utils.normalise_s2 import minmax_percentile





def plot_tensors(lr, sr, hr,title="Train",stretch=None):

    # prepare tensors
    if stretch=="sen2":
        sr = normalise_s2(sr,stage="denorm")
        lr = normalise_s2(lr,stage="denorm")
        hr = normalise_s2(hr,stage="denorm")
        lr = sen2_stretch(lr)
        sr = sen2_stretch(sr)
        hr = sen2_stretch(hr)
    elif stretch=="minmax":
        lr = minmax_percentile(lr)
        sr = minmax_percentile(sr)
        hr = minmax_percentile(hr)
    elif stretch==None:
        pass
    else:
        raise NotImplementedError("Stretch not implemented")

    sr = sr.clamp(0,1)
    lr = lr.clamp(0,1)
    hr = hr.clamp(0,1)


    B, _, W, H = lr.shape  # Assuming all tensors have the same shape except for possible W and H
    if B>5: # Restrict to Max 5 images
        B = 5
    
    fixed_width = 15
    variable_height = (15/3) * B
    fig, axes = plt.subplots(B, 3, figsize=(fixed_width, variable_height))
    
    # Loop over the batch size
    for i in range(B):
        # Extract individual images from the batch
        img_lr = lr[i].detach().cpu()
        img_sr = sr[i].detach().cpu()
        img_hr = hr[i].detach().cpu()

        # Plotting
        axes[i, 0].imshow(img_lr.permute(1,2,0).numpy())
        axes[i, 1].imshow(img_sr.permute(1,2,0).numpy())
        axes[i, 2].imshow(img_hr.permute(1,2,0).numpy())
        
        # Remove axis
        axes[i, 0].axis('off')
        axes[i, 0].set_title('LR')
        axes[i, 1].axis('off')
        axes[i, 1].set_title('SR')
        axes[i, 2].axis('off')
        axes[i, 2].set_title('HR')


     # Create a PIL image from the BytesIO object
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='png',dpi=150)
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close()

    # return PIL figure
    return pil_image

def plot_fusion(lr,enc,hr):

    # prepare tensors
    enc = normalise_s2(enc,stage="denorm")
    lr = normalise_s2(lr,stage="denorm")
    hr = normalise_s2(hr,stage="denorm")
    lr = sen2_stretch(lr)
    enc = sen2_stretch(enc)
    hr = sen2_stretch(hr)
    lr, enc, hr = torch.clamp(lr,0,1), torch.clamp(enc,0,1), torch.clamp(hr,0,1)

    # create fig and prepare tensors
    lr,enc,hr = lr[0],enc[0],hr[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    lr = lr.detach().cpu().numpy().transpose(1, 2, 0)
    enc = enc.detach().cpu().numpy().transpose(1, 2, 0)
    hr = hr.detach().cpu().numpy().transpose(1, 2, 0)

    # plot images
    axes[0].imshow(lr)
    axes[0].axis('off')
    axes[0].set_title('LR')
    axes[1].imshow(enc)
    axes[1].axis('off')
    axes[1].set_title('Fused Image')
    axes[2].imshow(hr)
    axes[2].axis('off')
    axes[2].set_title('HR')

    # save and return image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close()
    return pil_image

def misr_plot(lr,sr,hr):
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision

    # denorm etc
    hr = normalise_s2(hr,stage="denorm")
    hr = sen2_stretch(hr)
    sr = normalise_s2(sr,stage="denorm")
    sr = sen2_stretch(sr)
    lr = (((lr+1)/2)*10)/3
    lr = lr.clamp(0,1)
    sr = sr.clamp(0,1)

    # create LR grid
    lr = lr[0] # get first batch
    dim = lr.shape[-3] # get bands*views
    bands = 3 # get bands, hardcoded as 3
    images = int(dim/3) # get number of images
    #lr = lr.view(images,bands,lr.shape[-2], lr.shape[-1]) # reshape to go from V*B,W,H to V,B,W,H
    lr = torchvision.utils.make_grid(lr,nrow=2).cpu().numpy().transpose((1,2,0)) # make grid of images
    sr = sr[0].cpu().numpy().transpose((1,2,0))
    hr = hr[0].cpu().numpy().transpose((1,2,0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr)
    axes[0].axis('off')
    axes[0].set_title('LR')
    axes[1].imshow(sr)
    axes[1].axis('off')
    axes[1].set_title('SR')
    axes[2].imshow(hr)
    axes[2].axis('off')
    axes[2].set_title('HR')

    # save and return image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close()
    return pil_image

