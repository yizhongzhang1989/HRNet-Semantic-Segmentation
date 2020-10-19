import _init_paths
import os
import numpy as np
from PIL import Image
import datasets

root = 'data/panorama'
output_path = os.path.join(root, 'vis')
if not os.path.exists(output_path):
  os.makedirs(output_path)

panorama = datasets.panorama(root='data',
                             list_path=os.path.join(root, 'vis_list.txt'),
                             num_classes=5,
                             crop_size=(648, 864))

for i in range(10):
  for j in range(len(panorama)):
    image = panorama[j][0]
    PIL_image = Image.fromarray(image)  # transpose or not
    PIL_image.save(os.path.join(output_path, '{}_{}.png'.format(j, i)))

print('done')