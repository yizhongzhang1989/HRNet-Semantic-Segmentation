import os
import subprocess
import shutil
import numpy as np
import cv2
import random
from multiprocessing import Process

thread_num = 8
image_dir = "E:/StoreSemanticLabelingData"
output_dir = "E:/StoreSemanticTrainingData"


def list_pairs(dir):
    image_label_pair = []

    elems = os.listdir(dir)
    for elem in elems:
        elem_dir = '%s/%s' % (dir, elem)
        if os.path.isfile(elem_dir):  # a file is detected, check whether it is named *.jpg and *.png exists
            name, ext = elem_dir.split('.')
            if ext == 'jpg' or ext == 'JPG':
                image_dir = elem_dir
                label_dir = '%s.png' % name
                if os.path.exists(label_dir):
                    image_label_pair.append([image_dir, label_dir])

    return image_label_pair


def create_image_label(image_label_pairs, dst_image_dir, dst_label_dir):
    colormap = dict()
    with open("colorMap.txt", "r") as f:
        count = 0
        for line in f.readlines():
            tmp = line.split(" ")
            label = int(tmp[1])
            r, g, b = int(tmp[2]), int(tmp[3]), int(tmp[4])
            colormap[label] = np.array([b, g, r], dtype=np.int32)

    for pair in image_label_pairs:
        image, label = pair

        # copy image
        _, image_name = os.path.split(image)
        shutil.copyfile(image, os.path.join(dst_image_dir, image_name))

        # copy label
        segmentation = cv2.imread(label, cv2.IMREAD_COLOR)
        h, w = segmentation.shape[:2]
        label_img = np.ones([h, w], dtype=np.uint8) * 255
        for key in colormap:
            label_img[(segmentation == colormap[key]).all(axis=2)] = key
        _, label_name = os.path.split(label)
        cv2.imwrite(os.path.join(dst_label_dir, label_name), label_img)


if __name__ == '__main__':
    #   create directory
    if os.path.exists(output_dir):
        print('removed old ' + output_dir)
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    output_image_dir = os.path.join(output_dir, 'image')
    output_label_dir = os.path.join(output_dir, 'label')
    os.mkdir(output_image_dir)
    os.mkdir(output_label_dir)

    #   create training set
    pairs = list_pairs(image_dir)

    chunk_size = int(len(pairs) / thread_num)
    if chunk_size * thread_num != len(pairs):
        chunk_size += 1
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    for chunk in chunks:
        p = Process(target=create_image_label, args=(chunk, output_image_dir, output_label_dir))
        p.start()

    #   create list
    train_list = []
    test_list = []
    val_list = []

    for pair in pairs:
        image, _ = pair
        _, name = os.path.split(image)
        name, ext = name.split('.')

        tmp = random.randint(0, 6)
        if tmp == 0:
            val_list.append(name)
        elif tmp == 1:
            test_list.append(name)
        else:
            train_list.append(name)

    with open("%s/train_list.txt" % output_dir, "w") as f:
        f.writelines('\n'.join(train_list))
    with open("%s/test_list.txt" % output_dir, "w") as f:
        f.writelines('\n'.join(test_list))
    with open("%s/val_list.txt" % output_dir, "w") as f:
        f.writelines('\n'.join(val_list))
