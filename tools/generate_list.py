import random
import os
import cv2

data_dir = "E:/StoreSemanticLabelingData/merge"
total_list = []

for filename in os.listdir("%s/label" % data_dir):
    filename = filename.split(".")[0]
    label = cv2.imread("%s/label/%s.png" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
    invalid = (label == 255).mean()
    print(filename, ", %f percent of the pixels are invalid." % (invalid * 100))
    if invalid < 0.01:
        total_list.append(filename + "\n")

train_list = []
test_list = []
val_list = []

for line in total_list:
    tmp = random.randint(0, 6)
    if tmp == 0:
        val_list.append(line)
    elif tmp == 1:
        test_list.append(line)
    else:
        train_list.append(line)

with open("%s/train_list.txt" % data_dir, "w") as f:
    f.writelines(train_list)
with open("%s/test_list.txt" % data_dir, "w") as f:
    f.writelines(test_list)
with open("%s/val_list.txt" % data_dir, "w") as f:
    f.writelines(val_list)