import os
import cv2
import numpy as np

def crop_panorama_image(img, theta=0.0, phi=0.0, res_x=512, res_y=512, fov=60.0, NEAREST_INTER=False):
    img_x = img.shape[0]
    img_y = img.shape[1]

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi

    fov_x = fov
    aspect_ratio = res_y * 1.0 / res_x
    half_len_x = np.tan(fov_x / 180 * np.pi / 2)
    half_len_y = aspect_ratio * half_len_x

    pixel_len_x = 2 * half_len_x / res_x
    pixel_len_y = 2 * half_len_y / res_y

    map_x = np.zeros((res_x, res_y), dtype=np.float32)
    map_y = np.zeros((res_x, res_y), dtype=np.float32)

    axis_y = np.cos(theta)
    axis_z = np.sin(theta)
    axis_x = 0

    # theta rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    theta_rot_mat = np.array([[1, 0, 0], \
            [0, cos_theta, -sin_theta], \
            [0, sin_theta, cos_theta]], dtype=np.float32)

    # phi rotation matrix
    cos_phi = np.cos(phi)
    sin_phi = -np.sin(phi)
    phi_rot_mat = np.array([[cos_phi + axis_x**2 * (1 - cos_phi), \
            axis_x * axis_y * (1 - cos_phi) - axis_z * sin_phi, \
            axis_x * axis_z * (1 - cos_phi) + axis_y * sin_phi], \
            [axis_y * axis_x * (1 - cos_phi) + axis_z * sin_phi, \
            cos_phi + axis_y**2 * (1 - cos_phi), \
            axis_y * axis_z * (1 - cos_phi) - axis_x * sin_phi], \
            [axis_z * axis_x * (1 - cos_phi) - axis_y * sin_phi, \
            axis_z * axis_y * (1 - cos_phi) + axis_x * sin_phi, \
            cos_phi + axis_z**2 * (1 - cos_phi)]], dtype=np.float32)

    map_x = np.tile(np.array(np.arange(res_x), dtype=np.float32), (res_y, 1)).T
    map_y = np.tile(np.array(np.arange(res_y), dtype=np.float32), (res_x, 1))

    map_x = map_x * pixel_len_x + pixel_len_x / 2 - half_len_x
    map_y = map_y * pixel_len_y + pixel_len_y / 2 - half_len_y
    map_z = np.ones((res_x, res_y)).astype(np.float32) * -1

    ind = np.reshape(np.concatenate((np.expand_dims(map_x, 2), np.expand_dims(map_y, 2), \
            np.expand_dims(map_z, 2)), axis=2), [-1, 3]).T

    ind = theta_rot_mat.dot(ind)
    ind = phi_rot_mat.dot(ind)

    vec_len = np.sqrt(np.sum(ind**2, axis=0))
    ind /= np.tile(vec_len, (3, 1))

    cur_phi = np.arcsin(ind[0, :])
    cur_theta = np.arctan2(ind[1, :], -ind[2, :])

    map_x = (cur_phi + np.pi/2) / np.pi * img_x
    map_y = cur_theta % (2 * np.pi) / (2 * np.pi) * img_y

    map_x = np.reshape(map_x, [res_x, res_y])
    map_y = np.reshape(map_y, [res_x, res_y])

    if NEAREST_INTER:
        return cv2.remap(img, map_y, map_x, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    else:
        return cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

colormap = dict()
with open("colorMap.txt", "r") as f:
    for line in f.readlines():
        tmp = line.split(" ")
        label = int(tmp[1])
        r, g, b = int(tmp[2]), int(tmp[3]), int(tmp[4])
        colormap[label] = np.array([b,g,r], dtype=np.int32)

stores = ["store1", "store2", "store4&5", "store6", "store7", "store8", "store10", "store11"]

if not os.path.exists("D:/panorama/merge"):
    os.mkdir("D:/panorama/merge")
if not os.path.exists("D:/panorama/merge/image"):
    os.mkdir("D:/panorama/merge/image")
if not os.path.exists("D:/panorama/merge/label"):
    os.mkdir("D:/panorama/merge/label")

for store in stores:
    if not os.path.exists("D:/panorama/%s/label" % store):
        os.mkdir("D:/panorama/%s/label" % store)
    for filename in os.listdir("D:/panorama/%s" % store):
        filename = filename.split(".")
        if len(filename) != 2:
            continue
        if filename[1] == "jpg":
            filename = filename[0]
            if os.path.exists("D:/panorama/%s/label/%s.png" % (store, filename)):
                print("pass ", store, filename)
                continue
            print(store, filename)
            image = cv2.imread("D:/panorama/%s/%s.jpg" % (store, filename), cv2.IMREAD_COLOR)
            segmentation = cv2.imread("D:/panorama/%s/%s.png" % (store, filename), cv2.IMREAD_COLOR)

            h, w = segmentation.shape[:2]
            label = np.zeros([h, w], dtype=np.uint8)
            for key in colormap:
                label[(segmentation==colormap[key]).all(axis=2)] = key
            cv2.imwrite("D:/panorama/%s/label/%s.png" % (store, filename), label)
            #label = cv2.imread("D:/panorama/%s/label/%s.png" % (store, filename), cv2.IMREAD_GRAYSCALE)

            for i in range(6):
                theta = i * 60
                tmpImage = crop_panorama_image(image, theta, 10, 540, 720, 80)
                tmpLabel = crop_panorama_image(label, theta, 10, 540, 720, 80, True)
                cv2.imwrite("D:/panorama/merge/image/%s_%s_%d.jpg" % (store, filename, i), tmpImage)
                cv2.imwrite("D:/panorama/merge/label/%s_%s_%d.png" % (store, filename, i), tmpLabel)