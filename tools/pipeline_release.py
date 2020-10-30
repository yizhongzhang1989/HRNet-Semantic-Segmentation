import os
import argparse
import torch
from torch.nn import functional as F
import numpy as np
import cv2
from tqdm import tqdm

import _init_paths
import models
from config import config
from config import update_config


# args
parser = argparse.ArgumentParser(description='Train segmentation network')
parser.add_argument('--cfg', help='experiment configure file name',
                    type=str, default="experiments/panorama/train_v2.yaml")
parser.add_argument("--pth", help="pth file name", type=str,
                    default="output/panorama/1027_train_v2_scale_small/best.pth")
parser.add_argument("--input", help="input path", type=str,
                    default="data/panorama/key_images")
parser.add_argument("--output", help="output path", type=str,
                    default="data/panorama/key_images_output")
parser.add_argument("--scale_factor", help="scale factor to resize the image",
                    type=float, default=0.5)
parser.add_argument('opts', help="Modify config options using the command-line",
                    default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
update_config(config, args)


# input data
input_list, output_list = [], []
filenames = os.listdir(args.input)
if not os.path.exists(args.output):
  os.makedirs(args.output)
for filename in filenames:
  if filename.endswith('.jpg'):
    input_list.append(os.path.join(args.input, filename))
    output_list.append(os.path.join(args.output, filename[:-4] + '.png'))


# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.seg_hrnet.get_seg_model(config)
pretrained_dict = torch.load(args.pth, map_location=device)
model_dict = model.state_dict()
pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = model.to(device)
model.eval()


def resize_image(x, scale):
  h, w = int(scale * x.shape[0]), int(scale * x.shape[1])
  x = cv2.resize(x, (w, h))
  return x


def preprocess_function(x):
  x = np.array(x).astype(np.float32)
  x = x[:, :, ::-1]  # bgr -> rgb
  x = x / 255.0
  x -= [0.485, 0.456, 0.406]
  x /= [0.229, 0.224, 0.225]
  x = x.transpose([2, 0, 1])
  return x

# run
for i, image_name in enumerate(tqdm(input_list, ncols=80)):
  image = cv2.imread(image_name, cv2.IMREAD_COLOR)
  h, w = image.shape[0], image.shape[1]

  image = resize_image(image, args.scale_factor)
  image = preprocess_function(image)
  image = np.expand_dims(image, 0)
  image = torch.from_numpy(image).to(device)

  predict = model.forward(image)
  predict = F.upsample(predict, (h, w), mode='bilinear')
  predict = F.softmax(predict, dim=1)
  predict = predict.cpu().detach().numpy() # (1, C, H, W)

  label = np.argmax(predict, axis=1)
  label = np.squeeze(label).astype(np.uint8)
  confidence = np.max(predict, axis=1) * 255
  confidence = np.squeeze(confidence).astype(np.uint8)
  zeros = np.zeros_like(label)
  output = np.stack([label, confidence, zeros], axis=2)
  cv2.imwrite(output_list[i], output)
