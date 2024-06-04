import torch

def sen2_stretch(im):
    im = im*(10/3.)
    im = torch.clamp(im,0,1)
    return(im)