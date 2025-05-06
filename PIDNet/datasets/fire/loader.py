import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

class FireDataset(Dataset):
    def __init__(self, root, list_path, image_set='train', transform=None):
        self.root = root
        self.list_path = list_path
        self.image_set = image_set
        self.transform = transform

        with open(os.path.join(root, list_path), 'r') as f:
            self.img_list = [line.strip() for line in f.readlines()]

        self.image_dir = os.path.join(root, 'images', image_set)
        self.mask_dir = os.path.join(root, 'masks', image_set)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = self.convert_mask(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

    def convert_mask(self, mask_rgb):
        """
        Convert RGB mask to class index (0: background, 1: fire)
        based on your color mapping (red for fire, purple for background)
        """
        fire = [0, 0, 255]
        bg = [120, 20, 170]

        mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
        mask[np.all(mask_rgb == fire, axis=-1)] = 1  # Fire
        mask[np.all(mask_rgb == bg, axis=-1)] = 0    # Background
        return mask
