import os
import cv2
import numpy as np
import torch
from torch.nn import functional as F

def inference(network, arguments):
    input_list = arguments["input_list"]
    output_list = arguments["output_list"]
    if "preprocess_function" in arguments:
        preprocess_function = arguments["preprocess_function"]
    else:
        preprocess_function = lambda x: np.array(x).transpose([2,0,1]).astype(np.float32)

    for i, image_name in enumerate(input_list):
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        image = preprocess_function(image)
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image)
        predict = network.forward(image)
        predict = F.upsample(predict, (h, w), mode='bilinear')
        predict = F.softmax(predict, dim=1)
        predict = predict.cpu().detach().numpy()
        predict = np.argmax(predict, axis=1)
        predict = np.squeeze(predict).astype(np.uint8)
        if not os.path.exists(os.path.dirname(output_list[i])):
            os.makedirs(output_list[i])
        cv2.imwrite(output_list[i], predict)