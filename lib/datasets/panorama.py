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
        self.crop_size = crop_size
        self.list_path = list_path
        
        self.files = []
        self.images = []
        with open(list_path, "r") as f:
            for line in f.readlines():
                for _ in range(16):
                    self.files.append(line.strip("\n"))
                self.images.append(None)
    
    def __getitem__(self, index):
        if self.images[index // 16] is None:
            self.images[index // 16] = cv2.imread(os.path.join(self.root, "image", self.files[index] + ".jpg"), cv2.IMREAD_COLOR)
        image = self.images[index // 16]
        if 'val' in self.list_path:
            theta = np.random.random() * 360
            phi = np.random.random() * 20 - 10
        else:
            theta = random.random() * 360
            phi = random.random() * 20 - 10

        size = np.array([480, 480, 3])
        image = self.crop_panorama_image(image, theta, phi, 480, 480, 60)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        if 'test' in self.list_path:
            return image.copy(), np.array(size), self.files[index]
        else:
            label = cv2.imread(os.path.join(self.root, "label", self.files[index] + ".png"), cv2.IMREAD_GRAYSCALE)
            label = self.crop_panorama_image(label, theta, phi, 480, 480, 60)
            return image.copy(), label.copy(), np.array(size), self.files[index]
    
    def save_pred(self, preds, sv_path, name):
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            save_img = Image.fromarray(preds[i])
            save_img.save(os.path.join(sv_path, name[i]+'.png'))
    
    def crop_panorama_image(self, img, theta=0.0, phi=0.0, res_x=512, res_y=512, fov=60.0, debug=False):
        img_x = img.shape[0]
        img_y = img.shape[1]

        theta = theta / 180 * np.pi
        phi = phi / 180 * np.pi

        fov_x = fov
        aspect_ratio = res_y * 1.0 / res_x
        half_len_x = np.tan(fov_x / 180 * np.pi / 2)
        half_len_y = aspect_ratio * half_len_x

        pixel_len_x = 2 * half_len_x / res_x
        pixel_len_y = 2 * half_len_y / res_y

        map_x = np.zeros((res_x, res_y), dtype=np.float32)
        map_y = np.zeros((res_x, res_y), dtype=np.float32)

        axis_y = np.cos(theta)
        axis_z = np.sin(theta)
        axis_x = 0

        # theta rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        theta_rot_mat = np.array([[1, 0, 0], \
                [0, cos_theta, -sin_theta], \
                [0, sin_theta, cos_theta]], dtype=np.float32)

        # phi rotation matrix
        cos_phi = np.cos(phi)
        sin_phi = -np.sin(phi)
        phi_rot_mat = np.array([[cos_phi + axis_x**2 * (1 - cos_phi), \
                axis_x * axis_y * (1 - cos_phi) - axis_z * sin_phi, \
                axis_x * axis_z * (1 - cos_phi) + axis_y * sin_phi], \
                [axis_y * axis_x * (1 - cos_phi) + axis_z * sin_phi, \
                cos_phi + axis_y**2 * (1 - cos_phi), \
                axis_y * axis_z * (1 - cos_phi) - axis_x * sin_phi], \
                [axis_z * axis_x * (1 - cos_phi) - axis_y * sin_phi, \
                axis_z * axis_y * (1 - cos_phi) + axis_x * sin_phi, \
                cos_phi + axis_z**2 * (1 - cos_phi)]], dtype=np.float32)

        map_x = np.tile(np.array(np.arange(res_x), dtype=np.float32), (res_y, 1)).T
        map_y = np.tile(np.array(np.arange(res_y), dtype=np.float32), (res_x, 1))

        map_x = map_x * pixel_len_x + pixel_len_x / 2 - half_len_x
        map_y = map_y * pixel_len_y + pixel_len_y / 2 - half_len_y
        map_z = np.ones((res_x, res_y)).astype(np.float32) * -1

        ind = np.reshape(np.concatenate((np.expand_dims(map_x, 2), np.expand_dims(map_y, 2), \
                np.expand_dims(map_z, 2)), axis=2), [-1, 3]).T

        ind = theta_rot_mat.dot(ind)
        ind = phi_rot_mat.dot(ind)

        vec_len = np.sqrt(np.sum(ind**2, axis=0))
        ind /= np.tile(vec_len, (3, 1))

        cur_phi = np.arcsin(ind[0, :])
        cur_theta = np.arctan2(ind[1, :], -ind[2, :])

        map_x = (cur_phi + np.pi/2) / np.pi * img_x
        map_y = cur_theta % (2 * np.pi) / (2 * np.pi) * img_y

        map_x = np.reshape(map_x, [res_x, res_y])
        map_y = np.reshape(map_y, [res_x, res_y])

        if debug:
            for x in range(res_x):
                for y in range(res_y):
                    print("%.2f, %.2f)\t", map_x[x, y], map_y[x, y]),
                print

        return cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)