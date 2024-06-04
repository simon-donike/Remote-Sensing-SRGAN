import torch
from einops import rearrange


def normalise_s2(im,stage="norm"):
    # Selector for normalization process

    if len(im.shape)==5 and im.shape[2]==3: # we're in MISR with 3 bands, do 3 band older version linear transform
        return(normalise_s2_old(im,stage))

    elif len(im.shape)==4 and im.shape[1] == 3: # we're in 3 band SISR, do previous denorm
        return(normalise_s2_old(im,stage))
    
    elif len(im.shape)==4 and im.shape[1] == 4: # we're in 4 band MISR, return 4 band linear transform
        return linear_transform_4b(im,stage=stage)
    
    elif len(im.shape)==5 and im.shape[1] == 4: # we're in 4 band MISR, return 4 band linear transform
        return linear_transform_4b(im,stage=stage)
    
    else:
        raise NotImplementedError("Normalization not implemented for this shape of tensor")
    
    

def normalise_s2_old(im,stage="norm"):
    assert stage in ["norm","denorm"]
    value = 3.
    if stage == "norm":
        im = im*(10./value)
        im = (im*2)-1
        im = torch.clamp(im,-1,1)
    if stage=="denorm":
        im = (im+1)/2
        im = im*(value/10.)
        im = torch.clamp(im,0,1)
    return(im)


def linear_transform_4b(t_input,stage="norm"):
    assert stage in ["norm","denorm"]
    # get the shape of the tensor
    shape = t_input.shape

    # if 5 d tensor, norm/denorm individually
    if len(shape)==5:
        stack = []
        for batch in t_input:
            stack2 = []
            for i in range(0, t_input.size(1), 4):
                slice_tensor = batch[i:i+4, :, :, :]
                slice_denorm = linear_transform_4b(slice_tensor,stage=stage)
                stack2.append(slice_denorm)
            stack2 = torch.stack(stack2)
            stack2 = stack2.reshape(shape[1], shape[2], shape[3], shape[4])
            stack.append(stack2)
        stack = torch.stack(stack)
        return(stack)

    # here only if len(shape) == 4
    squeeze_needed = False
    if len( shape ) == 3:
        squeeze_needed = True
        t_input = t_input.unsqueeze(0)
        shape = t_input.shape

    assert len(shape)==4 or len(shape)==5,"Input tensor must have 4 dimensions (B,C,H,W) - or 5D for MISR"
    transpose_needed = False
    if shape[-1]>shape[1]:
        transpose_needed = True
        t_input = rearrange(t_input,"b c h w -> b w h c")
    
    # define constants
    rgb_c = 3.
    nir_c = 5.

    # iterate over batches
    return_ls = []
    for t in t_input:
        if stage == "norm":
            # divide according to conventions
            t[:,:,0] = t[:,:,0] * (10.0 / rgb_c) # R
            t[:,:,1] = t[:,:,1] * (10.0 / rgb_c) # G
            t[:,:,2] = t[:,:,2] * (10.0 / rgb_c) # B
            t[:,:,3] = t[:,:,3] * (10.0 / nir_c) # NIR
            # clamp to get rif of outlier pixels
            t = t.clamp(0,1)
            # bring to -1..+1
            t = (t*2)-1
        if stage == "denorm":
            # bring to 0..1
            t = (t+1)/2
            # divide according to conventions
            t[:,:,0] = t[:,:,0] * (rgb_c / 10.0) # R
            t[:,:,1] = t[:,:,1] * (rgb_c / 10.0) # G
            t[:,:,2] = t[:,:,2] * (rgb_c / 10.0) # B
            t[:,:,3] = t[:,:,3] * (nir_c / 10.0) # NIR
            # clamp to get rif of outlier pixels
            t = t.clamp(0,1)
        
        # append result to list
        return_ls.append(t)

    # after loop, stack image
    t_output = torch.stack(return_ls)
    #print("stacked",t_output.shape)

    if transpose_needed==True:
        t_output = rearrange(t_output,"b w h c -> b c h w")
    if squeeze_needed:
        t_output = t_output.squeeze(0)

    return(t_output)