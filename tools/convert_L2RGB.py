import cv2
import random
import os
import numpy as np

colormap = dict()
with open("colorMap.txt", "r") as f:
    for line in f.readlines():
        tmp = line.split(" ")
        label = int(tmp[1])
        r, g, b = int(tmp[2]), int(tmp[3]), int(tmp[4])
        colormap[label] = np.array([r,g,b], dtype=np.int32)

for filename in os.listdir("D:/testNet/result"):
    filename = filename.split(".")[0]
    store, number = filename.split("_")

    if os.path.exists("D:/%s/label/%s.png" % (store, filename)):
        ground_truth = cv2.imread("D:/%s/label/%s.png" % (store, filename), cv2.IMREAD_GRAYSCALE)
        h, w = ground_truth.shape
        tmp = np.zeros([h, w, 3], dtype=np.uint8)
        for key in colormap:
            tmp[ground_truth==key] = colormap[key]
        tmp = tmp[:,:,::-1]
        cv2.imwrite("D:/testNet/convert/%s_truth.png" % filename, tmp)

    predict = cv2.imread("D:/testNet/result/%s.png" % filename, cv2.IMREAD_GRAYSCALE)
    h, w = predict.shape
    tmp = np.zeros([h, w, 3], dtype=np.uint8)
    for key in colormap:
        tmp[predict==key] = colormap[key]
    tmp = tmp[:,:,::-1]
    cv2.imwrite("D:/testNet/convert/%s_predict.png" % filename, tmp)