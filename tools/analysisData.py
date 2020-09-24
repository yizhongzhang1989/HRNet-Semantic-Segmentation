import os
import numpy as np
import cv2

data_dir = "D:/panorama/merge"

def analysis(listfile, label_map):
    total = []
    with open(listfile, "r") as f:
        filelist = f.readlines()
    for filename in filelist:
        filename = filename.strip("\n")
        print(filename)
        count = np.zeros([41], dtype=np.int64)
        label = cv2.imread("%s/label/%s.png" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
        for i in label_map:
            tmp = (label == i).sum()
            count[i] += tmp
        total.append(count)
    return np.array(total)

label_map = dict()
with open("colorMap.txt", "r") as f:
    for line in f.readlines():
        tmp = line.split(" ")
        classname = tmp[0]
        label = int(tmp[1])
        label_map[label] = classname

validate_result = analysis("%s/val_list.txt" % data_dir, label_map)
test_result = analysis("%s/test_list.txt" % data_dir, label_map)
train_result = analysis("%s/train_list.txt" % data_dir, label_map)

for key in label_map:
    print(label_map[key], key, validate_result.mean(axis=0)[key], test_result.mean(axis=0)[key], train_result.mean(axis=0)[key], validate_result.var(axis=0)[key], test_result.var(axis=0)[key], train_result.var(axis=0)[key])