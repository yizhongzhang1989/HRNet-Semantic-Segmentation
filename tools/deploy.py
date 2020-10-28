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
  parser.add_argument('--cfg', help='experiment configure file name',
                      type=str, default="experiments/panorama/train.yaml")
  parser.add_argument("--pth", help="pth file name", type=str,
                      default="output/panorama/1021_train_c17/best.pth")
  parser.add_argument('opts', help="Modify config options using the command-line",
                      default=None, nargs=argparse.REMAINDER)
  args = parser.parse_args()
  update_config(config, args)
  return args


# load the model
args = parse_args()
model = models.seg_hrnet.get_seg_model(config)
pretrained_dict = torch.load(args.pth)
model_dict = model.state_dict()
pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = model.cuda()
model.eval()


# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
example = torch.rand(1, 3, 800, 600).cuda()
output = model.forward(example)
# https://github.com/pytorch/vision/issues/2161
# m = torch.jit.script(model)

traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("traced_hrnet_model.pt")

