import os
import argparse
import torch

import _init_paths
import models
from config import config
from config import update_config
from utils.inference import inference, resize_image, default_preprocess_function

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--scale_factor",
                        default=0.6,
                        help="scale factor to resize the image",
                        type=float)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

args = parse_args()
model = models.seg_hrnet.get_seg_model(config)
pretrained_dict = torch.load("output/panorama/train/best.pth")
model_dict = model.state_dict()

pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = model.cuda()
model.eval()

# test_name = 'data/panorama/test_list.txt'
# with open(test_name, 'r') as f:
#     test_list = f.readlines()
# input_list = ['data/panorama/image/%s.jpg' % name[:-1] for name in test_list]
# output_list = ['data/panorama/predict/%s.png' % name[:-1] for name in test_list]

test_list = os.listdir('data/panorama/key_images')
input_list = ['data/panorama/key_images/%s.jpg' % name[:-4] for name in test_list if name.endswith('.jpg')]
output_list = ['data/panorama/key_images_predict_scale0.6/%s.png' % name[:-4] for name in test_list if name.endswith('.jpg')]
os.makedirs('data/panorama/key_images_predict_scale0.6')


def preprocess_function(x):
    x = resize_image(x, args.scale_factor)
    x = default_preprocess_function(x)
    return x


inference(model,
         {"input_list": input_list,
          "output_list": output_list,
          "preprocess_function": preprocess_function})