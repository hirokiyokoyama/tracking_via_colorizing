import cv2
import numpy as np
import os
from itertools import product

_data_dir = os.path.join(os.path.dirname(__file__), 'data')
_davis_dir = os.path.join(_data_dir, 'DAVIS_trainval')
_annotations_dir = os.path.join(_davis_dir, 'Annotations', '480p')
_images_dir = os.path.join(_davis_dir, 'JPEGImages', '480p')
if not os.path.exists(_davis_dir):
    print 'DAVIS dataset not found. Downloading.'
    _url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip'
    _zip_file = os.path.join(_data_dir, 'DAVIS-2017-trainval-480p.zip')
    os.system('curl "%s" > %s' % (_url, _zip_file))
    os.system('unzip %s -d %s' % (_zip_file, _data_dir))
    os.remove(_zip_file)

def get_image_sets(year='2017', name='train'):
    image_sets_file = os.path.join(_davis_dir, 'ImageSets', year, name+'.txt')
    return map(lambda x: x.strip(), open(image_sets_file).readlines())

def get_annotations(image_set):
    img_dir = os.path.join(_annotations_dir, image_set)
    img_files = os.listdir(img_dir)
    img_files.sort()

    imgs = np.array([cv2.imread(os.path.join(img_dir, f)) for f in img_files])
    imgs32 = np.uint32(imgs[:,:,:,0]) | np.uint32(imgs[:,:,:,1]) << 8 | np.uint32(imgs[:,:,:,2]) << 16
    masks = np.zeros(imgs.shape[:3], np.uint8)
    color_dict = {0:0}
    colors = [np.array([0,0,0], np.uint8)]
    count = 1
    for i, j in product(xrange(imgs.shape[1]), xrange(imgs.shape[2])):
        pixel = imgs32[0,i,j]
        if pixel in color_dict:
            masks[0,i,j] = color_dict[pixel]
        else:
            masks[0,i,j] = count
            color_dict[pixel] = count
            colors.append(imgs[0,i,j])
            count += 1
    for color32, i in color_dict.items():
        if i != 0:
            masks[1:][imgs32[1:] == color32] = i
    return masks, np.array(colors)

def get_images(image_set):
    img_dir = os.path.join(images_dir, image_set)
    img_files = os.listdir(img_dir)
    img_files.sort()
    return np.array([cv2.imread(os.path.join(img_dir, f)) for f in img_files])
