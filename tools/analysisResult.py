import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

analysis_simplified = False
analysis_compressed = True
data_dir = "D:/panorama/merge"
result_dir = "D:/testNet/result"
output_dir = "D:/testNet"

compress_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    20: 14,
    30: 15,
    40: 16,
}

colormap = dict()
with open("colorMap.txt", "r") as f:
    for line in f.readlines():
        tmp = line.split(" ")
        label = int(tmp[1])
        r, g, b = int(tmp[2]), int(tmp[3]), int(tmp[4])
        if analysis_compressed:
            colormap[compress_map[label]] = np.array([r,g,b], dtype=np.int32)
        else:
            colormap[label] = np.array([r,g,b], dtype=np.int32)

def plot_confusion_matrix(cm):
    import matplotlib.pyplot as plt
    label_map = {
            0: "ground",
            1: "ceiling",
            2: "wall",
            3: "pillar",
            4: "door",
            5: "window",
            6: "stairs",
            7: "escalator",
            8: "elevator",
            10: "layered_shelf",
            11: "table_shelf",
            13: "tall_freezer",
            14: "short_freezer",
            15: "cashier",
            20: "hanging_object",
            30: "other_object",
            40: "removed_object"
        }
    if cm.shape[0] == 41:
        class_names = []
        slices = []
        for i in range(41):
            if i in label_map:
                slices.append(True)
                class_names.append(label_map[i])
            else:
                slices.append(False)
        cm = cm[slices]
        cm = cm[:,slices]
    elif cm.shape[0] == 17:
        class_names = []
        for key in label_map:
            class_names.append(label_map[key])
    
    figure = plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2.
    normalized = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, normalized[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

if analysis_simplified:
    matrix = np.zeros([4, 4], dtype=np.int64)
elif analysis_compressed:
    matrix = np.zeros([17, 17], dtype=np.int64)
else:
    matrix = np.zeros([41, 41], dtype=np.int64)

all_results = os.listdir(result_dir)
errors = []
for filename in all_results:
    #print(filename)
    predict = cv2.imread("%s/%s" % (result_dir, filename), cv2.IMREAD_GRAYSCALE)
    if analysis_simplified:
        truth = cv2.imread("%s/simplify/%s" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
    elif analysis_compressed:
        truth = cv2.imread("%s/compress/%s" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
    else:
        truth = cv2.imread("%s/label/%s" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
    h, w = predict.shape
    error = 0
    error = (predict != truth).sum()
    errors.append(error)

figure = plot_confusion_matrix(matrix)
plt.savefig("%s/analysis.png" % output_dir)

errors = np.array(errors)

for i in np.argsort(-errors)[:20]:
    filename = all_results[i]
    name = filename.split(".")[0]
    image = cv2.imread("%s/image/%s.jpg" % (data_dir, name), cv2.IMREAD_COLOR)
    predict = cv2.imread("%s/%s" % (result_dir, filename), cv2.IMREAD_GRAYSCALE)
    if analysis_simplified:
        truth = cv2.imread("%s/simplify/%s" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
    elif analysis_compressed:
        truth = cv2.imread("%s/compress/%s" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
    else:
        truth = cv2.imread("%s/label/%s" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
    h, w = truth.shape
    RGB_label = np.zeros([h, w, 3], dtype=np.uint8)
    for key in colormap:
        RGB_label[truth==key] = colormap[key]
    RGB_label = RGB_label[:,:,::-1]
    mixed_truth = cv2.addWeighted(image, 0.3, RGB_label, 0.7, 0)
    cv2.imwrite("%s/hard/%s_truth.jpg" % (output_dir, name), mixed_truth)
    for key in colormap:
        RGB_label[predict==key] = colormap[key]
    RGB_label = RGB_label[:,:,::-1]
    cv2.imwrite("%s/hard/%s_predict.jpg" % (output_dir, name), RGB_label)

for i in np.argsort(errors)[:20]:
    filename = all_results[i]
    name = filename.split(".")[0]
    image = cv2.imread("%s/image/%s.jpg" % (data_dir, name), cv2.IMREAD_COLOR)
    predict = cv2.imread("%s/%s" % (result_dir, filename), cv2.IMREAD_GRAYSCALE)
    if analysis_simplified:
        truth = cv2.imread("%s/simplify/%s" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
    elif analysis_compressed:
        truth = cv2.imread("%s/compress/%s" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
    else:
        truth = cv2.imread("%s/label/%s" % (data_dir, filename), cv2.IMREAD_GRAYSCALE)
    h, w = truth.shape
    RGB_label = np.zeros([h, w, 3], dtype=np.uint8)
    for key in colormap:
        RGB_label[truth==key] = colormap[key]
    RGB_label = RGB_label[:,:,::-1]
    mixed_truth = cv2.addWeighted(image, 0.3, RGB_label, 0.7, 0)
    cv2.imwrite("%s/easy/%s_truth.jpg" % (output_dir, name), mixed_truth)
    for key in colormap:
        RGB_label[predict==key] = colormap[key]
    RGB_label = RGB_label[:,:,::-1]
    cv2.imwrite("%s/easy/%s_predict.jpg" % (output_dir, name), RGB_label)