import argparse
import torch
import cv2
import numpy as np

import _init_paths
import models
from config import config
from config import update_config
from utils.inference import default_preprocess_function as preprocess
from utils.inference import batch_inference as inference

colormap = dict()
compress_map = dict()
with open("colorMap.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        tmp = line.split(" ")
        label = int(tmp[1])
        r, g, b = int(tmp[2]), int(tmp[3]), int(tmp[4])
        colormap[label] = np.array([r,g,b], dtype=np.int32)
        compress_map[label] = i

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg', help='experiment configure file name', type=str, default="experiments/panorama/train.yaml")
    parser.add_argument("--pth", help="pth file name", type=str, default="D:/testNet/final_state.pth")
    parser.add_argument("--input_video", help="input video file name", type=str, default="D:/testNet/input.mp4")
    parser.add_argument("--output_video", help="output video file name, should be .avi format", type=str, default="D:/testNet/output.avi")
    parser.add_argument("--batch_size", help="frames per batch", type=int, default=12)
    parser.add_argument("--scale_factor", help="scale factor to resize the image", type=float, default=0.5)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def process_batch(network, batch, original):
    predicts, probabilities = inference(model, batch, output_probability=True)
    for i, label in enumerate(predicts):
        result = np.zeros((h, w, 3), dtype=np.uint8)
        probability = np.zeros((h, w, 3), dtype=np.float32)
        probability[:,:] = [255, 0, 255]
        probability[:,:,0] *= 1 - probabilities[i]
        probability[:,:,2] *= probabilities[i]
        probability = probability.astype(np.uint8)
        for key in compress_map:
            result[label == compress_map[key]] = colormap[key]
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
while inputVideo.isOpened():
    ret, frame = inputVideo.read()
    if ret is False:
        break
    count += 1
    print(count, "/", total_frame)
    frame = cv2.resize(frame, (w, h))
    #cv2.imshow("frame", frame)
    #if cv2.waitKey(1) == ord('q'):
    #    break
    batch.append(preprocess(frame))
    original.append(frame)
    if len(batch) == batch_size:
        process_batch(model, batch, original)
        batch = []
        original = []

if len(batch) > 0:
    process_batch(model, batch, original)

inputVideo.release()
outputVideo.release()
cv2.destroyAllWindows()