import os

data_dir = "E:/StorePanoramaSemanticLabelingData"


def query_dir(dir):
    image_label_pair = []

    elems = os.listdir(dir)
    for elem in elems:
        elem_dir = '%s/%s' % (dir, elem)
        if os.path.isdir(elem_dir):     #   a sub directory, query deeper
            image_label_pair = image_label_pair + query_dir(elem_dir)
        elif os.path.isfile(elem_dir):  #   a file is detected, check whether it is named *.jpg and *.png exists
            name, ext = elem_dir.split('.')
            if ext == 'jpg' or ext == 'JPG':
                image_dir = elem_dir
                label_dir = '%s.png' % name
                if os.path.exists(label_dir):
                    image_label_pair.append([image_dir, label_dir])

    return image_label_pair



image_label_pair_list = query_dir(data_dir)
output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../bin/panorama_image_label_list.txt')

with open(output_path, 'w') as f:
    for pair in image_label_pair_list:
        f.write('%s %s\n' % (pair[0], pair[1]))

    print('write %d image pairs' % len(image_label_pair_list))