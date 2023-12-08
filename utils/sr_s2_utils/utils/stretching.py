import torch

def convention_stretch_sen2(t):
    # assuming range of t=0..1
    # times 10000 to get to the Sen2 range and then /3000 by convention:
    # https://github.com/google/dynamicworld/blob/master/single_image_runner.ipynb
    t = t * (10 / 3)
    t = t.clamp(0,1)
    return(t)

def hq_histogram_matching(
        image1: torch.Tensor, image2: torch.Tensor
    ) -> torch.Tensor:
        """Lazy implementation of histogram matching

        Args:
            image1 (torch.Tensor): The super-resolved image (C, H, W).
            image2 (torch.Tensor): The low-resolution (C, H, W).

        Returns:
            torch.Tensor: The super-resolved image with the histogram of
                the target image.
        """
        from skimage.exposure import match_histograms
        import torch
        import numpy
        # add check for multi-batches
        req_squeeze = False
        if len(image1.shape)==3:
            image1 = image1.unsqueeze(0)
            image2 = image2.unsqueeze(0)
            req_squeeze = True
        
        out_ts_ls = []
        for image1_,image2_ in zip(image1,image2):
            # Go to numpy
            image1_ = image1_.detach().cpu().numpy()
            image2_ = image2_.detach().cpu().numpy()

            if image1_.ndim == 3:
                np_image1_hat = match_histograms(image1_, image2_, channel_axis=0)
            elif image1_.ndim == 2:
                np_image1_hat = match_histograms(image1_, image2_, channel_axis=None)
            else:
                raise ValueError("The input image must have 2 or 3 dimensions.")

            # Go back to torch
            image1_hat = torch.from_numpy(np_image1_hat)
            out_ts_ls.append(image1_hat)

        # stack out tensors
        image1_hat = torch.stack(out_ts_ls).to(image1.device)
        
        if req_squeeze:
            image1_hat = image1_hat.squeeze(0)

        return image1_hat
    

    