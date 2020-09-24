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

compress_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    20: 14,
    30: 15,
    40: 16,
}

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

colormap = dict()
with open("colorMap.txt", "r") as f:
    for line in f.readlines():
        tmp = line.split(" ")
        label = int(tmp[1])
        r, g, b = int(tmp[2]), int(tmp[3]), int(tmp[4])
        colormap[label] = np.array([r,g,b], dtype=np.uint8)

inputVideo = cv2.VideoCapture(input_video_dir)
outputVideo = None

original = []
batch = []
while inputVideo.isOpened():
    ret, frame = inputVideo.read()
    if ret is False:
        break
    h, w = frame.shape[:2]
    h, w = h * scale_factor, w * scale_factor
    h, w = int(h), int(w)
    frame = cv2.resize(frame, (w, h))
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    if outputVideo is None:
        outputVideo = cv2.VideoWriter(output_video_dir, cv2.VideoWriter_fourcc(*"DIVX"), inputVideo.get(cv2.CAP_PROP_FPS), (w * 2, h * 2))
    if len(batch) < batch_size:
        batch.append(preprocess(frame))
        original.append(frame)
    else:
        process_batch(model, batch, original)
        batch = []
        original = []

if len(batch) > 0:
    process_batch(model, batch, original)

inputVideo.release()
outputVideo.release()
cv2.destroyAllWindows()