import argparse
import torch
import cv2
import numpy as np
from threading import Thread
from multiprocessing import Queue

import _init_paths
import models
from config import config
from config import update_config
from utils.inference import default_preprocess_function as preprocess
from utils.inference import batch_inference as inference

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg', help='experiment configure file name', type=str, default="D:/testNet/small_v2_train.yaml")
    parser.add_argument("--pth", help="pth file name", type=str, default="D:/testNet/final_state.pth")
    parser.add_argument("--input_video", help="input video file name", type=str, default="D:/testNet/input1.mp4")
    parser.add_argument("--output_video", help="output video file name, should be .avi format", type=str, default="D:/testNet/optical_flow1.avi")
    parser.add_argument("--batch_size", help="frames per batch", type=int, default=4)
    parser.add_argument("--scale_factor", help="scale factor to resize the image", type=float, default=0.71111111111111111111)
    parser.add_argument("--sliding_window", type=int, default=5)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def producer(queue, args, config):
    scale_factor = args.scale_factor
    model = models.seg_hrnet.get_seg_model(config)
    pretrained_dict = torch.load(args.pth)
    model_dict = model.state_dict()

    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.cuda()

    inputVideo = cv2.VideoCapture(args.input_video)
    total_frame = int(inputVideo.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(inputVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(inputVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = inputVideo.get(cv2.CAP_PROP_FPS)

    h, w = h * scale_factor, w * scale_factor
    h, w = int(h), int(w)
    queue.put([total_frame, h, w, fps])
    count = 0
    while inputVideo.isOpened():
        ret, frame = inputVideo.read()
        if ret is False:
            break
        frame = cv2.resize(frame, (w, h))
        _, _, raws = inference(model, [preprocess(frame)], output_max_probability=True, output_raw=True)
        raws = raws[0].transpose(1, 2, 0)
        queue.put([raws, frame])
        count += 1
    inputVideo.release()
    queue.put(None)

def consumer(queue, args, compressedColorMap):
    total_frame, h, w, fps = queue.get()
    outputVideo = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"DIVX"), fps, (w * 2, h * 2))

    map_x = np.zeros([h,w], dtype=np.float32)
    map_y = np.zeros([h,w], dtype=np.float32)
    for i in range(h):
        map_y[i,:] = i
    for i in range(w):
        map_x[:,i] = i

    raws_window = []
    frames_window = []
    flows_window = []
    flow = None
    count = 0
    while True:
        data = queue.get()
        if not data:
            outputVideo.release()
            break
        count += 1
        print(count, "/", total_frame)
        raw, frame = data
        if len(frames_window) >= 1:
            flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frames_window[-1], cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), flow, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow[:,:,0] += map_x
            flow[:,:,1] += map_y
            flows_window.append(flow)
        frames_window.append(frame)

        this_raws = [raw]
        for flow in flows_window[args.sliding_window-1::-1]:
            raw = cv2.remap(raw, flow[:,:,0], flow[:,:,1], cv2.INTER_LINEAR)
            this_raws.append(raw)
        raws_window.append(this_raws)

        if len(frames_window) >= args.sliding_window:
            this_frames_window = frames_window[:args.sliding_window]
            this_raws_window = raws_window[:args.sliding_window]

            label = np.argmax(this_raws_window[0][-1], axis=2)
            original_result = np.zeros((h, w, 3), dtype=np.uint8)
            for key in compressedColorMap:
                original_result[label == key] = compressedColorMap[key]

            averaged_raw = np.sum([this_raws_window[i][-1] for i in range(args.sliding_window)], axis=0)
            averaged_label = np.argmax(averaged_raw, axis=2)
            averaged_result = np.zeros((h, w, 3), dtype=np.uint8)
            for key in compressedColorMap:
                averaged_result[averaged_label == key] = compressedColorMap[key]
            
            compare = np.zeros((2*h, 2*w, 3), dtype=np.uint8)
            compare[:h,:w,:] = original_result[:,:,::-1]
            compare[:h,w:,:] = averaged_result[:,:,::-1]
            compare[h:,:w,:] = this_frames_window[0]
            frames_window.pop(0)
            raws_window.pop(0)
            for raws in raws_window:
                raws.pop(-1)
            flows_window.pop(0)
            outputVideo.write(compare)

if __name__ == '__main__':
    compressedColorMap = dict()
    with open("compressMap.txt", "r") as f:
        for i, line in enumerate(f.readlines()):
            tmp = line.split(" ")
            r, g, b = int(tmp[1]), int(tmp[2]), int(tmp[3])
            compressedColorMap[i] = np.array([r,g,b], dtype=np.int32)

    args = parse_args()

    queue = Queue(20)
    p = Thread(target=producer, args=(queue, args, config))
    c = Thread(target=consumer, args=(queue, args, compressedColorMap))

    p.start()
    c.start()