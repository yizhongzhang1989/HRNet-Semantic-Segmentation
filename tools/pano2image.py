import os
import subprocess
from multiprocessing import Process

thread_num = 8
panorama_dir = "E:/StorePanoramaSemanticLabelingData"
output_dir = "E:/StoreSemanticLabelingData"
samples_per_panorama = 12
image_width = 1024
image_height = 768
fov_min = 90.0
fov_max = 107.0
rand_tilt = True
rand_ground = True


def quote_string(in_str):
    if in_str == '':
        return ''
    if in_str[0] == '"' and in_str[-1] == '"':
        return in_str

    if detect_special_chars(in_str) != '':
        return '"' + in_str + '"'

    return in_str


def detect_special_chars(in_str):
    out_special_chars = ''

    special_chars = [' ', '&']
    for c in special_chars:
        if c in in_str:
            out_special_chars = out_special_chars + c

    return out_special_chars


def list_panorama_pairs(dir):
    """
    Recursively list all panorama image-label pairs in a directory

    Return full path of [[image1, label1], [image2, label2], [image3, label3], ... ]
    """
    image_label_pair = []

    elems = os.listdir(dir)
    for elem in elems:
        elem_dir = '%s/%s' % (dir, elem)
        if os.path.isdir(elem_dir):  # a sub directory, query deeper
            image_label_pair = image_label_pair + list_panorama_pairs(elem_dir)
        elif os.path.isfile(elem_dir):  # a file is detected, check whether it is named *.jpg and *.png exists
            name, ext = elem_dir.split('.')
            if ext == 'jpg' or ext == 'JPG':
                image_dir = elem_dir
                label_dir = '%s.png' % name
                if os.path.exists(label_dir):
                    image_label_pair.append([image_dir, label_dir])

    return image_label_pair


def save_panorama_pairs(image_label_pair, output_dir, chunk_num=thread_num):
    """
    Save image_label list into files
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_filenames = []
    output_pair_num = []

    chunk_size = int(len(image_label_pair) / chunk_num)
    if chunk_size * chunk_num != len(image_label_pair):
        chunk_size += 1
    chunks = [image_label_pair[i:i + chunk_size] for i in range(0, len(image_label_pair), chunk_size)]

    for idx in range(len(chunks)):
        chunk = chunks[idx]
        output_path = '%s/panorama_image_label_%d.txt' % (output_dir, idx)
        with open(output_path, 'w') as f:
            for pair in chunk:
                f.write('%s %s\n' % (pair[0], pair[1]))
            print('write %s with %d pairs' % (output_path, len(chunk)))
            f.close()
            output_filenames.append(output_path)
            output_pair_num.append(len(chunk))

    return output_filenames, output_pair_num


def pano_2_images(pano_list_filename,
                  output_dir,
                  start_index=0,
                  samples_per_panorama=12,
                  image_width=1024,
                  image_height=768,
                  fov_min= 90.0,
                  fov_max=107.0,
                  rand_tilt=True,
                  rand_ground=True):
    cmd = 'Pano2Image'
    cmd = cmd + ' -input_panorama_image_label_filename ' + quote_string(pano_list_filename)
    cmd = cmd + ' -output_dir ' + output_dir
    cmd = cmd + ' -start_index %d' % start_index
    cmd = cmd + ' -samples_per_panorama %d' % samples_per_panorama
    cmd = cmd + ' -image_width %d' % image_width
    cmd = cmd + ' -image_height %d' % image_height
    cmd = cmd + ' -fov_min %f' % fov_min
    cmd = cmd + ' -fov_max %f' % fov_max
    if not rand_tilt:
        cmd = cmd + ' -rand_tilt false'
    if not rand_ground:
        cmd = cmd + ' -rand_ground false'
    print(cmd)
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    list = list_panorama_pairs(panorama_dir)
    output_filenames, output_pair_num = save_panorama_pairs(list, output_dir, thread_num)

    acc_pair_num = 0
    for i in range(len(output_filenames)):
        start_idx = acc_pair_num *  samples_per_panorama

        p = Process(target=pano_2_images, args=(output_filenames[i],
                                                output_dir,
                                                start_idx,
                                                samples_per_panorama,
                                                image_width,
                                                image_height,
                                                fov_min,
                                                fov_max,
                                                rand_tilt,
                                                rand_ground))
        p.start()

        acc_pair_num += output_pair_num[i]
