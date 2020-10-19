import argparse
import torch

import _init_paths
import models
from config import config
from config import update_config
from utils.inference import inference

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

args = parse_args()
model = models.seg_hrnet.get_seg_model(config)
pretrained_dict = torch.load("output/panorama/train/final_state.pth")
model_dict = model.state_dict()

pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = model.cuda()


test_name = 'data/panorama/test_list.txt'
with open(test_name, 'r') as f:
    test_list = f.readlines()

input_list = ['data/panorama/image/%s.jpg' % name[:-1] for name in test_list]
output_list = ['data/panorama/predict/%s.png' % name[:-1] for name in test_list]

inference(model, {"input_list": input_list, "output_list": output_list})