import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
import torchvision.transforms as transforms
from utils.sen2_stretch import sen2_stretch
from utils.normalise_s2 import normalise_s2





def plot_tensors(lr, sr, hr,title="Train"):

    # prepare tensors
    sr = normalise_s2(sr,stage="denorm")
    lr = normalise_s2(lr,stage="denorm")
    hr = normalise_s2(hr,stage="denorm")
    lr = sen2_stretch(lr)
    sr = sen2_stretch(sr)
    hr = sen2_stretch(hr)
    lr, sr, hr = torch.clamp(lr,0,1), torch.clamp(sr,0,1), torch.clamp(hr,0,1)


    B, _, W, H = lr.shape  # Assuming all tensors have the same shape except for possible W and H
    
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
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close()

    # return PIL figure
    return pil_image
