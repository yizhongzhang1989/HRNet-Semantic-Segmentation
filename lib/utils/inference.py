import os
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

def default_preprocess_function(x):
    x = np.array(x).astype(np.float32)
    x = x[:,:,::-1]
    x = x / 255.0
    x -= [0.485, 0.456, 0.406]
    x /= [0.229, 0.224, 0.225]
    x = x.transpose([2,0,1])
    return x

def single_image_inference(network, image):
    network.eval()

    h, w = image.shape[1:]
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).cuda()
    predict = network.forward(image)
    predict = F.upsample(predict, (h, w), mode='bilinear')
    predict = F.softmax(predict, dim=1)
    predict = predict.cpu().detach().numpy()
    predict = np.argmax(predict, axis=1)
    predict = np.squeeze(predict).astype(np.uint8)
    return predict

def batch_inference(network, batch, output_max_probability=False, output_raw=False):
    network.eval()

    h, w = batch[0].shape[1:]
    batch = np.array(batch)
    batch = torch.from_numpy(batch).cuda()
    predict = network.forward(batch)
    predict = F.upsample(predict, (h, w), mode='bilinear')
    raw = predict
    predict = F.softmax(predict, dim=1)
    predict = predict.cpu().detach().numpy()
    label = np.argmax(predict, axis=1).astype(np.uint8)
    if output_max_probability:
        max_probability = np.max(predict, axis=1)
        if not output_raw:
            return label, max_probability
        else:
            return label, max_probability, raw.cpu().detach().numpy()
    else:
        return label

def inference(network, arguments):
    input_list = arguments["input_list"]
    output_list = arguments["output_list"]
    if "preprocess_function" in arguments:
        preprocess_function = arguments["preprocess_function"]
    else:
        preprocess_function = default_preprocess_function

    for i, image_name in enumerate(tqdm(input_list, ncols=80)):
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = preprocess_function(image)
        predict = single_image_inference(network, image)
        if not os.path.exists(os.path.dirname(output_list[i])):
            os.makedirs(output_list[i])
        cv2.imwrite(output_list[i], predict)