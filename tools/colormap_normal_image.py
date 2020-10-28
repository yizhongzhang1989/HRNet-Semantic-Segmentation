import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

image_dir = "data/Spatial_Reconstruction/normal_image"
predict_dir = "data/Spatial_Reconstruction/normal_image_predict"
output_dir = "data/Spatial_Reconstruction/normal_image_output"


# filenames
folders = os.listdir(image_dir)
image_list, predict_list, output_list = [], [], []
for folder in folders:
  folder_image = os.path.join(image_dir, folder)
  folder_predict = os.path.join(predict_dir, folder)
  folder_output = os.path.join(output_dir, folder)
  if not os.path.isdir(folder_image): continue
  if not os.path.exists(folder_output): os.makedirs(folder_output)

  filenames = os.listdir(folder_image)
  for filename in filenames:
    if filename.endswith('.jpg'):
      image_list.append(os.path.join(folder_image, filename))
      predict_list.append(os.path.join(folder_predict, filename[:-4] + '.png'))
      output_list.append(os.path.join(folder_output, filename[:-4] + '.png'))


# load compressedColorMap
compressMap = dict()
compressedColorMap = dict()
with open("compressMap.txt", "r") as f:
  for i, line in enumerate(f.readlines()):
    tmp = line.split(" ")
    r, g, b = int(tmp[1]), int(tmp[2]), int(tmp[3])
    compressedColorMap[i] = np.array([r, g, b], dtype=np.int32)
    for label in tmp[4:]:
      compressMap[int(label)] = i


# generate image
accu = []
for i in tqdm(range(len(image_list)), ncols=80):
  image = cv2.imread(image_list[i], cv2.IMREAD_COLOR)
  label_rgb_gt = cv2.imread(image_list[i][:-4] + '.png', cv2.IMREAD_COLOR)
  predict = cv2.imread(predict_list[i], cv2.IMREAD_GRAYSCALE)

  h, w = predict.shape
  label_rgb = np.zeros([h, w, 3], dtype=np.uint8)
  for key in compressedColorMap:
    label_rgb[predict == key] = compressedColorMap[key]
  label_rgb[:, :, ::-1] = label_rgb  # rgb -> bgr

  mixed = np.zeros([h, w * 2, 3], dtype=np.uint8)
  mixed[:, :w, :] = cv2.addWeighted(image, 0.5, label_rgb_gt, 0.5, 0)
  mixed[:, w:, :] = label_rgb
  cv2.imwrite(output_list[i], mixed)

  accu.append((label_rgb_gt == label_rgb).prod(-1).mean())

# save accu
with open(os.path.join(output_dir, 'accu.csv'), 'w') as fid:
  for i in range(len(image_list)):
    fid.write('%s, %f\n' % (image_list[i], accu[i]))
  fid.write('All, %f\n' % np.array(accu).mean())