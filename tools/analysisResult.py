import os
import cv2
import numpy as np
import shutil

data_dir = "data/panorama"
result_dir = "data/panorama/predict"
output_dir = "data/panorama/output"
topK = 20

compressMap = dict()
compressedColorMap = dict()
with open("compressMap.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        tmp = line.split(" ")
        r, g, b = int(tmp[1]), int(tmp[2]), int(tmp[3])
        compressedColorMap[i] = np.array([r,g,b], dtype=np.int32)
        for label in tmp[4:]:
            compressMap[int(label)] = i

if os.path.exists("%s/easy" % output_dir):
    shutil.rmtree("%s/easy" % output_dir)   
if os.path.exists("%s/hard" % output_dir):
    shutil.rmtree("%s/hard" % output_dir)
os.makedirs("%s/easy" % output_dir)
os.makedirs("%s/hard" % output_dir)

all_results = os.listdir(result_dir)
errors = []
for filename in all_results:
    print(filename)
    predict = cv2.imread("%s/%s" % (result_dir, filename), cv2.IMREAD_GRAYSCALE)
    truth = cv2.imread("%s/label/%s" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
    h, w = predict.shape
    error = 0
    for key in compressMap:
        if key != compressMap[key]:
            truth[truth == key] = compressMap[key]
    error = (predict != truth).sum()
    errors.append(error)

errors = np.array(errors)

for i in np.argsort(-errors)[:topK]:
    filename = all_results[i]
    name = filename.split(".")[0]
    image = cv2.imread("%s/image/%s.jpg" % (data_dir, name), cv2.IMREAD_COLOR)
    predict = cv2.imread("%s/%s" % (result_dir, filename), cv2.IMREAD_GRAYSCALE)
    truth = cv2.imread("%s/label/%s" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
    h, w = truth.shape
    RGB_label = np.zeros([h, w, 3], dtype=np.uint8)
    mixed = np.zeros([h, w * 2, 3], dtype=np.uint8)
    for key in compressMap:
        RGB_label[truth==key] = compressedColorMap[compressMap[key]]
    mixed[:,:w,:] = cv2.addWeighted(image, 0.3, RGB_label[:,:,::-1], 0.7, 0)
    for key in compressedColorMap:
        RGB_label[predict==key] = compressedColorMap[key]
    mixed[:,w:,:] = RGB_label[:,:,::-1]
    cv2.imwrite("%s/hard/%s_compare.png" % (output_dir, name), mixed)

for i in np.argsort(errors)[:topK]:
    filename = all_results[i]
    name = filename.split(".")[0]
    image = cv2.imread("%s/image/%s.jpg" % (data_dir, name), cv2.IMREAD_COLOR)
    predict = cv2.imread("%s/%s" % (result_dir, filename), cv2.IMREAD_GRAYSCALE)
    truth = cv2.imread("%s/label/%s" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
    h, w = truth.shape
    RGB_label = np.zeros([h, w, 3], dtype=np.uint8)
    mixed = np.zeros([h, w * 2, 3], dtype=np.uint8)
    for key in compressMap:
        RGB_label[truth==key] = compressedColorMap[compressMap[key]]
    mixed[:,:w,:] = cv2.addWeighted(image, 0.3, RGB_label[:,:,::-1], 0.7, 0)
    for key in compressedColorMap:
        RGB_label[predict==key] = compressedColorMap[key]
    mixed[:,w:,:] = RGB_label[:,:,::-1]
    cv2.imwrite("%s/easy/%s_compare.png" % (output_dir, name), mixed)