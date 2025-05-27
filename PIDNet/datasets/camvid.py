# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import os
import numpy as np
from PIL import Image
import torch
import pilgram 
import cv2

from .base_dataset import BaseDataset

class CamVid(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 transform=None,
                 saturate=False,
                 num_classes=2,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=1280, 
                 crop_size=(720, 1280),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(CamVid, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.transform = transform
        self.saturate = saturate

        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path) if line.strip()]

        self.files = self.read_files()

        self.ignore_label = ignore_label
        
        self.color_list = [[170,40,120], [255, 0, 0], ]
        
        self.class_weights = None
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []

        for item in self.img_list:
            if len(item) < 2:
                continue
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
            
        return files
        
    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2])*self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2)==3] = i

        return label.astype(np.uint8)
    
    def label2color(self, label):
        color_map = np.zeros(label.shape+(3,))
        for i, v in enumerate(self.color_list):
            color_map[label==i] = self.color_list[i]
            
        return color_map.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image_path = os.path.join(self.root, item["img"])
        label_path = os.path.join(self.root, item["label"])

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.saturate:
            edit = np.random.randn(1)
            if edit < 0.5:
                factor = 1 + 0.09 * torch.randn(1).item()
                image = pilgram.css.contrast(image, factor)
            else:
                factor = 1 + 0.09 * torch.randn(1).item()
                image = pilgram.css.saturate(image, factor)

        image = np.array(image)
        size = image.shape

        color_map = Image.open(label_path).convert('RGB')
        if self.transform:
            color_map = self.transform(color_map)
        color_map = np.array(color_map)
        label = self.color2label(color_map)
        label[label == 255] = 0  # Αντικαθιστά το ignore label με background


        # Debugging info
        #print("\n==== Debug Sample ====\n")
        #print("Sample Name:", name)
        #print("Image path:", image_path)
        #print("Label path:", label_path)
        #print("Image shape:", image.shape)
        #print("Label shape:", label.shape)
        #print("Label unique values:", np.unique(label))
        #print("=======================\n")

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, city=False)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

