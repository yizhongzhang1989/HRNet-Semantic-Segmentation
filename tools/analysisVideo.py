import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm

import _init_paths
import models
from config import config
from config import update_config
from utils.inference import default_preprocess_function as preprocess
from utils.inference import batch_inference as inference

compressedColorMap = dict()
with open("compressMap.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        tmp = line.split(" ")
        r, g, b = int(tmp[1]), int(tmp[2]), int(tmp[3])
        compressedColorMap[i] = np.array([r,g,b], dtype=np.int32)

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg', help='experiment configure file name', type=str, default="experiments/panorama/train.yaml")
    parser.add_argument("--pth", help="pth file name", type=str, default="output/panorama/1021_train_c17/best.pth")
    parser.add_argument("--input_video", help="input video file name", type=str, default="data/panorama/20200505_14-05-32-250.mp4")
    parser.add_argument("--output_video", help="output video file name, should be .avi format", type=str, default="data/panorama/20200505_14-05-32-250-output-0.5.avi")
    parser.add_argument("--batch_size", help="frames per batch", type=int, default=16)
    parser.add_argument("--scale_factor", help="scale factor to resize the image", type=float, default=0.5)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def process_batch(network, batch, original, outputVideo):
    predicts, probabilities = inference(model, batch, output_max_probability=True)
    for i, label in enumerate(predicts):
        result = np.zeros((h, w, 3), dtype=np.uint8)
        probability = np.zeros((h, w, 3), dtype=np.float32)
        probability[:,:] = [255, 0, 255]
        probability[:,:,0] *= 1 - probabilities[i]
        probability[:,:,2] *= probabilities[i]
        probability = probability.astype(np.uint8)
        for key in compressedColorMap:
            result[label == key] = compressedColorMap[key]
        result = result[:,:,::-1]
        compare = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        compare[:h,:w,:] = original[i]
        compare[h:,:w,:] = cv2.addWeighted(original[i], 0.5, probability, 0.5, 0)
        compare[:h,w:,:] = result
        compare[h:,w:,:] = cv2.addWeighted(original[i], 0.3, result, 0.7, 0)
        outputVideo.write(compare)

args = parse_args()
input_video_dir = args.input_video
output_video_dir = args.output_video
batch_size = args.batch_size
scale_factor = args.scale_factor
model = models.seg_hrnet.get_seg_model(config)
pretrained_dict = torch.load(args.pth)
model_dict = model.state_dict()

pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = model.cuda()
model.eval()

inputVideo = cv2.VideoCapture(input_video_dir)
total_frame = int(inputVideo.get(cv2.CAP_PROP_FRAME_COUNT))
h = int(inputVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(inputVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = inputVideo.get(cv2.CAP_PROP_FPS)

h, w = h * scale_factor, w * scale_factor
h, w = int(h), int(w)
outputVideo = cv2.VideoWriter(output_video_dir, cv2.VideoWriter_fourcc(*"DIVX"), fps, (w * 2, h * 2))

original = []
batch = []
count = 0
pbar = tqdm(total=total_frame, ncols=80)
while inputVideo.isOpened():
    ret, frame = inputVideo.read()
    if ret is False:
        break
    count += 1
    pbar.update(1)
    frame = cv2.resize(frame, (w, h))
    batch.append(preprocess(frame))
    original.append(frame)
    if len(batch) == batch_size:
        process_batch(model, batch, original, outputVideo)
        batch = []
        original = []
pbar.close()

# if len(batch) > 0:
#     process_batch(model, batch, original, outputVideo)

inputVideo.release()
outputVideo.release()
cv2.destroyAllWindows()