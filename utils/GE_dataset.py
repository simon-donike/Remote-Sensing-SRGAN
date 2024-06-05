import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from einops import rearrange

class ImageDataset(Dataset):
    def __init__(self, root_dir,phase="train"):
        self.hr_dir = os.path.join(root_dir,phase,"HR")
        self.lr_dir = os.path.join(root_dir,phase,"LR")
        self.hr_images = sorted(os.listdir(self.hr_dir))
        self.lr_images = sorted(os.listdir(self.lr_dir))
        self.common_images = sorted(list(set(self.hr_images) & set(self.lr_images)))

        if phase == "train":
            self.augmentations = A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.0),
                A.GaussianBlur(blur_limit=(3, 3), p=0.0),
                A.RandomBrightnessContrast(p=0.0)
            ])
        else:
            self.augmentations = None

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        im_p = self.common_images[idx]
        hr_image_path = os.path.join(self.hr_dir, im_p)
        lr_image_path = os.path.join(self.lr_dir, im_p)
        #print(hr_image_path,lr_image_path)

        hr_image = Image.open(hr_image_path).convert("RGB")
        lr_image = Image.open(lr_image_path).convert("RGB")

        hr_image = np.array(hr_image)
        lr_image = np.array(lr_image)

        if self.augmentations:
            augmented = self.augmentations(image=hr_image, image0=lr_image)
            hr_image = augmented['image']
            lr_image = augmented['image0']

        # turn images into torch
        hr_image = torch.Tensor(hr_image)/255.
        lr_image = torch.Tensor(lr_image)/255.
        hr_image = rearrange(hr_image, 'h w c -> c h w')
        lr_image = rearrange(lr_image, 'h w c -> c h w')

        return {"image":hr_image, "LR_image":lr_image}


if __name__ == "__main__":
    # Example usage
    root_dir = "/data1/simon/datasets/vietnam_google"
    dataset = ImageDataset(root_dir ,phase="train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Iterate through the dataloader
    for x in tqdm(dataloader):
        print(x["image"].shape, x["LR_image"].shape)

if __name__ == "__main__":
    
    def plot_images_side_by_side(image1, image2, title1='Image 1', title2='Image 2',name="test"):
        import matplotlib.pyplot as plt
        """
        Plots two images side by side.
        
        Args:
            image1 (numpy array): First image to plot.
            image2 (numpy array): Second image to plot.
            title1 (str): Title for the first image.
            title2 (str): Title for the second image.
        """
        # Transpose images from CHW to HWC format
        image1 = np.transpose(image1, (1, 2, 0))
        image2 = np.transpose(image2, (1, 2, 0))

        # Create a figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the first image
        axes[0].imshow(image1)
        axes[0].set_title(title1)
        axes[0].axis('off')

        # Plot the second image
        axes[1].imshow(image2)
        axes[1].set_title(title2)
        axes[1].axis('off')

        # Display the images
        plt.tight_layout()
        plt.savefig("images/"+str(name)+".png")

    for i in range(10):
        ims = dataset[i]
        lr,hr = ims["image"], ims["LR_image"]
        plot_images_side_by_side(lr,hr,title1="LR",title2="HR",name=i)

