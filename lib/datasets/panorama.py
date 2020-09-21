import os

import cv2
import numpy as np
import random
from PIL import Image

import torch

from .base_dataset import BaseDataset

class Panorama(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=2,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=800, 
                 crop_size=(480, 480), 
                 downsample_rate=1,
                 scale_factor=16,
                 center_crop_test=False,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],):
    
        super(Panorama, self).__init__(ignore_label, base_size,
                    crop_size, downsample_rate, scale_factor, mean, std)
        
        self.root = os.path.join(root, "panorama")

        self.num_classes = num_classes
        self.class_weights = None
        self.multi_scale = multi_scale
        self.flip = flip
        self.center_crop_test = center_crop_test
        self.crop_size = crop_size
        self.list_path = list_path
        
        self.files = []
        with open(list_path, "r") as f:
            for line in f.readlines():
                self.files.append(line.strip("\n"))
    
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.root, "image", self.files[index] + ".jpg"), cv2.IMREAD_COLOR)
        size = image.shape
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        if 'test' in self.list_path:
            return image.copy(), np.array(size), self.files[index]
        else:
            label = cv2.imread(os.path.join(self.root, "label", self.files[index] + ".png"), cv2.IMREAD_GRAYSCALE)
            image, label = self.gen_sample(image, label, self.multi_scale, self.flip, self.center_crop_test)
            return image.copy(), label.copy(), np.array(size), self.files[index]
    
    def save_pred(self, preds, sv_path, name):
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            save_img = Image.fromarray(preds[i])
            save_img.save(os.path.join(sv_path, name[i]+'.png'))