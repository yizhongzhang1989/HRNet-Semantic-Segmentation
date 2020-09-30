import os
import subprocess

thread_num = 8
panorama_dir = "E:/StorePanoramaSemanticLabelingData"
output_dir = "E:/StoreSemanticLabelingData2"


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

    return output_filenames


def pano_2_images(pano_list_filename,
                  output_dir,
                  start_idx=0,
                  samples_per_panorama=12,
                  image_width=1042,
                  image_height=768,
                  fov_min= 90.0,
                  fov_max=107.0,
                  rand_tilt=True,
                  rand_ground=True):
    """

    """
    curr_path = os.path.dirname(os.path.realpath(__file__))
    root_path = curr_path + '/..'

    cmd = 'cmd /c PATH=%%PATH%%;%s/3rdParty/bin/win64;%s/3rdParty/bin/win32;%s/bin/Release' % (root_path, root_path, root_path)
    cmd = cmd + ' && echo %%PATH%% &&  Pano2Image'
    #print(cmd)
    subprocess.call(cmd, shell=True)



#list = list_panorama_pairs(panorama_dir)
#save_panorama_pairs(list, output_dir, thread_num)
pano_2_images('a', 'b')

