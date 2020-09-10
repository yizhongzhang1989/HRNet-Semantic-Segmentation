import random
import os

data_dir = "D:/panorama/merge/image"
total_list = []

for filename in os.listdir(data_dir):
    filename = filename.split(".")[0]
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

with open("train_list.txt", "w") as f:
    f.writelines(train_list)
with open("test_list.txt", "w") as f:
    f.writelines(test_list)
with open("val_list.txt", "w") as f:
    f.writelines(val_list)