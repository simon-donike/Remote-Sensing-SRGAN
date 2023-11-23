import torch

def sen2_stretch(im):
    im = im*(10/3.)
    return(im)