# cifar10rotation.py

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image
import torchvision.transforms.functional as TF



def rotate_img(img, rot):
    """Rotate an image by 0, 90, 180, or 270 degrees."""
    # print(rot)
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 1:  # 90 degrees rotation
        return torch.rot90(img, 1, [1, 2])
    elif rot == 2:  # 180 degrees rotation
        return torch.rot90(img, 2, [1, 2])  # Rotate 90 degrees twice
    elif rot == 3:  # 270 degrees rotation
        return torch.rot90(img, 3, [1, 2])  # Rotate 90 degrees three times
    else:
        raise ValueError('rotation should be 0, 1, 2, or 3')



class CIFAR10Rotation(torchvision.datasets.CIFAR10):

    def __init__(self, root, train, download, transform) -> None:
        super().__init__(root=root, train=train, download=download, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        image, cls_label = super().__getitem__(index)
        # randomly select image rotation
        rotation_label = random.choice([0, 1, 2, 3])
        image_rotated = rotate_img(image, rotation_label)


        rotation_label = torch.tensor(rotation_label).long()
        return image, image_rotated, rotation_label, torch.tensor(cls_label).long()
