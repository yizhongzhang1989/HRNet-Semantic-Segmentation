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

for filename in os.listdir("D:/panorama/result-72/image"):
    filename = filename.split(".")[0]
    labeled = cv2.imread("D:/panorama/result-72/%s.png" % filename, cv2.IMREAD_COLOR)
    labeled = labeled[:,:,::-1]

    h, w, _ = labeled.shape
    tmp = np.zeros([h, w], dtype=np.uint8)
    for key in colormap:
        tmp[(labeled==colormap[key]).all(axis=2)] = key
    print(filename)
    cv2.imwrite("D:/panorama/result-72/label/%s.png" % filename, tmp)