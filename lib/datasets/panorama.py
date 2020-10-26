import os

import cv2
import numpy as np
import random
from PIL import Image

import torch
import torchvision.transforms as transforms

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
                 std=[0.229, 0.224, 0.225]):
    
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
        
        self.compress_map = dict()
        with open("compressMap.txt", "r") as f:
            for i, line in enumerate(f.readlines()):
                tmp = line.split(" ")
                for label in tmp[4:]:
                    self.compress_map[int(label)] = i
        
        image_size = (768, 1024)
        self.barrel_distortion_map = self.initialize_barrel_distortion_map(image_size)
        self.motion_blur_kernels = np.array([
            [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]],
            [[0,1,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,1,0]],
            [[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
            [[0,0,0,1,0],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[0,1,0,0,0]],
            [[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0]],
            [[0,0,0,0,0],[0,0,0,1,1],[0,0,1,0,0],[1,1,0,0,0],[0,0,0,0,0]],
            [[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0]],
            [[0,0,0,0,0],[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1],[0,0,0,0,0]]
        ], dtype=np.float32) / 5
    
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.root, "image", self.files[index] + ".jpg"), cv2.IMREAD_COLOR)
        image = self.bgr2rgb(image)
        size = image.shape
        if 'test' in self.list_path:            
            image = self.normalize_image(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), np.array(size), self.files[index]
        else:
            color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2) # only supports PIL or tensor input
            label = cv2.imread(os.path.join(self.root, "label", self.files[index] + ".png"), cv2.IMREAD_GRAYSCALE)
            label = self.compress_label(label)
            image = self.random_motion_blur(image)
            image, label = self.barrel_distortion(image, label)
            
            # image, label = self.rand_crop(image, label)
            rand_scale = 0.8 + random.randint(0, 6) / 10.0
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)
            
            image = np.asarray(color_jitter(Image.fromarray(image)))            
            image = self.normalize_image(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), label.copy(), np.array(size), self.files[index]
    
    def bgr2rgb(self, image):
        return image[:, :, ::-1]

    def normalize_image(self, image):
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def save_pred(self, preds, sv_path, name):
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            save_img = Image.fromarray(preds[i])
            save_img.save(os.path.join(sv_path, name[i]+'.png'))
    
    def compress_label(self, label):
        for key in self.compress_map:
            if key != self.compress_map[key]:
                label[label == key] = self.compress_map[key]
        return label
    
    def random_motion_blur(self, image):
        choice = random.randint(0, 10)
        if choice >= 8:
            return image
        else:
            return cv2.filter2D(image, -1, self.motion_blur_kernels[choice])
    
    def barrel_distortion(self, image, label):
        map_x, map_y = self.barrel_distortion_map
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR), cv2.remap(label, map_x, map_y, cv2.INTER_NEAREST)

    def initialize_barrel_distortion_map(self, image_size):
        h, w = image_size
        center_x = (w - 1) / 2
        center_y = (h - 1) / 2

        map_x = np.ones((h,w), dtype=np.float32) * center_x
        map_y = np.ones((h,w), dtype=np.float32) * center_y

        k1 = -0.3
        k2 = 0.3

        f = lambda r_square, k1, k2: 1 - k1 * r_square + (3 * k1 ** 2 - k2) * r_square ** 2

        max_r_square = center_x ** 2 + center_y ** 2
        fit_scale = f(1, k1, k2)

        for y in range(h):
            for x in range(w):
                r_square = (x - center_x) ** 2 + (y - center_y) ** 2
                r_square /= max_r_square
                scale = f(r_square, k1, k2)
                map_x[y][x] += (x - center_x) * scale / fit_scale
                map_y[y][x] += (y - center_y) * scale / fit_scale
        
        return [map_x, map_y]