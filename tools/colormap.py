import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

image_dir = "data/panorama/key_images"
predict_dir = "data/panorama/key_images_predict_scale0.5"
output_dir = "data/panorama/key_images_output_scale0.5"
scale_factor = 0.5


if not os.path.exists(output_dir):
  os.makedirs(output_dir)

compressMap = dict()
compressedColorMap = dict()
with open("compressMap.txt", "r") as f:
  for i, line in enumerate(f.readlines()):
    tmp = line.split(" ")
    r, g, b = int(tmp[1]), int(tmp[2]), int(tmp[3])
    compressedColorMap[i] = np.array([r, g, b], dtype=np.int32)
    for label in tmp[4:]:
      compressMap[int(label)] = i


all_results = os.listdir(predict_dir)
for i in tqdm(range(len(all_results)), ncols=80):
  filename = all_results[i]
  image = cv2.imread(os.path.join(image_dir, filename[:-3] + 'jpg'), cv2.IMREAD_COLOR)
  h, w = int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor)
  image = cv2.resize(image, (w, h))
  predict = cv2.imread(os.path.join(predict_dir, filename), cv2.IMREAD_GRAYSCALE)

  h, w = predict.shape
  RGB_label = np.zeros([h, w, 3], dtype=np.uint8)
  for key in compressedColorMap:
    RGB_label[predict == key] = compressedColorMap[key]

  mixed = np.zeros([h, w * 2, 3], dtype=np.uint8)
  mixed[:,:w,:] = image
  mixed[:, w:,::-1] = RGB_label  
  # mixed = np.concatenate([image, cv2.cvtColor(RGB_label, cv2.COLOR_RGB2BGR)], axis=1)  
  cv2.imwrite(os.path.join(output_dir, filename), mixed)