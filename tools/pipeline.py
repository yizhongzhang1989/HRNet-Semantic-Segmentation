import os
import argparse
import torch

import _init_paths
import models
from config import config
from config import update_config
from utils.inference import inference, resize_image, default_preprocess_function

# ## normal_image
input_list, output_list = [], []
root_folder = 'data/Spatial_Reconstruction/normal_image'
folders = os.listdir(root_folder)
for folder in folders:
  folder_input = os.path.join(root_folder, folder)
  folder_output = os.path.join(root_folder + '_predict', folder)
  if not os.path.isdir(folder_input): continue
  if not os.path.exists(folder_output): os.makedirs(folder_output)

  filenames = os.listdir(folder_input)
  for filename in filenames:
    if filename.endswith('.jpg'):
      input_list.append(os.path.join(folder_input, filename))
      output_list.append(os.path.join(folder_output, filename[:-4] + '.png'))


# test filelist
# test_name = 'data/panorama/test_list.txt'
# with open(test_name, 'r') as f:
#     test_list = f.readlines()
# input_list = ['data/panorama/image/%s.jpg' % name[:-1] for name in test_list]
# output_list = ['data/panorama/predict/%s.png' % name[:-1] for name in test_list]

# ## for yizhong
# test_list = os.listdir('data/panorama/key_images')
# input_list = ['data/panorama/key_images/%s.jpg' % name[:-4] for name in test_list if name.endswith('.jpg')]
# output_list = ['data/panorama/key_images_predict_scale0.6/%s.png' % name[:-4] for name in test_list if name.endswith('.jpg')]
# os.makedirs('data/panorama/key_images_predict_scale0.6')


def parse_args():
  parser = argparse.ArgumentParser(description='Train segmentation network')
  parser.add_argument('--cfg', help='experiment configure file name',
                      type=str, default="experiments/panorama/train.yaml")
  parser.add_argument("--pth", help="pth file name", type=str,
                      default="output/panorama/1021_train_c17/best.pth")
  parser.add_argument("--scale_factor", help="scale factor to resize the image",
                      type=float, default=0.5)
  parser.add_argument('opts', help="Modify config options using the command-line",
                      default=None, nargs=argparse.REMAINDER)
  args = parser.parse_args()
  update_config(config, args)
  return args


args = parse_args()
model = models.seg_hrnet.get_seg_model(config)
pretrained_dict = torch.load(args.pth)
model_dict = model.state_dict()

pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = model.cuda()
model.eval()


def preprocess_function(x):
  x = resize_image(x, args.scale_factor)
  x = default_preprocess_function(x)
  return x


def postprocess_function(x):
  x = resize_image(x, 1.0 / args.scale_factor)
  return x

inference(model, input_list, output_list, preprocess_function, postprocess_function)