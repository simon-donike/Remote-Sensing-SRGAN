import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from skimage import exposure


"""
GENERAL
"""
def plot_tensor(im):
    pass

def minmax(img):
    min_val = torch.min(img)
    max_val = torch.max(img)
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img

"""
SPECTRAL MATCHING METHODS
"""
# Normal Standardization over whole dataset
def normalize(sen2,spot6,sen2_amount=1):
    transform_spot = transforms.Compose([transforms.Normalize(mean=[479.0, 537.0, 344.0], std=[430.0, 290.0, 229.0]) ])
    # dynamically define transform to reflect shape of tensor
    trans_mean,trans_std = [78.0, 91.0, 62.0]*sen2_amount,[36.0, 28.0, 30.0]*sen2_amount
    transform_sen = transforms.Compose([transforms.Normalize(mean=trans_mean, std= trans_std)])
    # perform transform
    sen2  = transform_sen(sen2)
    spot6 = transform_spot(spot6)
    return sen2,spot6

# HISTOGRAM MATCHING
def histogram(sen2,spot6,sen2_amount=None):
    # https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.match_histograms
    # have to transpose so that multichannel understands the dimensions
    sen2,spot6 = sen2.cpu().numpy(),spot6.cpu().numpy() # turn to np from tensor
    sen2 = np.transpose(sen2,(1,2,0))
    spot6 = np.transpose(spot6,(1,2,0))
    spot6 = exposure.match_histograms(image=spot6,reference=sen2,channel_axis=2)
    spot6,sen2 = np.transpose(spot6,(2,0,1)),np.transpose(sen2,(2,0,1))
    spot6,sen2 = torch.Tensor(spot6),torch.Tensor(sen2)
    return spot6

# MOMENT MATCHING
def moment(sen2,spot6,sen2_amount=None):   
    sen2,spot6 = sen2.numpy(),spot6.numpy()
    c = 0
    for channel_sen,channel_spot in zip(sen2,spot6):
        c +=1
        #calculate stats
        sen2_mean   = np.mean(channel_sen)
        spot6_mean  = np.mean(channel_spot)
        sen2_stdev  = np.std(channel_sen)
        spot6_stdev = np.std(channel_spot)

        # calculate moment per channel
        channel_result = (((channel_spot - spot6_mean) / spot6_stdev) * sen2_stdev) + sen2_mean

        # stack channels to single array
        if c==1:
            spot6 = channel_result
        else:
            spot6 = np.dstack((spot6,channel_result))
        # transpose back to Cx..

    spot6 = torch.Tensor(spot6.transpose((2,0,1)))   
    return spot6 


"""
SPATIAL MATCHING METHODS
"""
def grid(lr=None,hr=None):
    return(hr)
def shiftnet(lr=None,hr=None):
    return(hr)

"""
Define Identity
"""
def identity(lr,hr):
    return(lr)