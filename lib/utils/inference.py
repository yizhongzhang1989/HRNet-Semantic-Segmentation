import cv2
import numpy as np
from torch.nn import functional as F

def inference(network, arguments):
    input_list = arguments["input_list"]
    output_list = arguments["output_list"]
    if "preprocess_function" in arguments:
        preprocess_function = arguments["preprocess_function"]
    else:
        preprocess_function = lambda x: np.array(x).astype(np.float32)

    for i, image_name in enumerate(input_list):
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = preprocess_function(image)
        image = np.expand_dims(image, 0)
        predict = network.forward(image)
        predict = F.softmax(predict, dim=1)
        predict = predict.cpu().numpy()
        predict = np.squeeze(predict)
        cv2.imwrite(output_list[i], predict)