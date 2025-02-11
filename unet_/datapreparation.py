from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms, io
from PIL import Image

torchvision_transforms = transforms

class UnetDataPreparation(Dataset):
    def __init__(self, ImagesDirectory, MasksDirectory):
        self.ImagesDirectory = ImagesDirectory
        self.MasksDirectory = MasksDirectory
        self.images = sorted(os.listdir(self.ImagesDirectory))
        self.masks = sorted(os.listdir(self.MasksDirectory))

        self.mask_transforms = torchvision_transforms.Compose([
            torchvision_transforms.Resize((512, 512), interpolation=torchvision_transforms.InterpolationMode.NEAREST)
        ])

        self.image_transforms = torchvision_transforms.Compose([
            torchvision_transforms.ToPILImage(),
            torchvision_transforms.Resize((512, 512), interpolation=torchvision_transforms.InterpolationMode.NEAREST),
            torchvision_transforms.ToTensor(),
            torchvision_transforms.Lambda(lambda x: x[0:3])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get the image filename
        img_name = self.images[index]

        # Construct the path to the image
        img_path = os.path.join(self.ImagesDirectory, img_name)

        # Read and decode the image
        img = io.read_file(img_path)
        img = io.decode_png(img)

        # Generate the corresponding mask filename
        mask_name = img_name.replace(' ', '_').replace('.PNG', '_mask.PNG')  # Replace spaces with underscores if necessary

        # Construct the path to the mask
        mask_path = os.path.join(self.MasksDirectory, mask_name)

        # Read the mask image
        mask = io.read_image(mask_path)

        # Apply the transformations to both the image and the mask
        img, mask = self.image_transforms(img), self.mask_transforms(mask)
        mask_recode_dict = {170: 3, 85: 2, 0: 0, 255: 1} 

        recoded_mask = torch.zeros_like(mask, dtype=torch.long)
        for k, v in mask_recode_dict.items():
            recoded_mask[mask == k] = v

        return img, recoded_mask
