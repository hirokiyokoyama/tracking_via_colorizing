#!/usr/bin/env python3

import tensorflow as tf
import cv2
import numpy as np
import os
from nets import create_feature_extractor, Colorizer
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'model')
DAVIS_DIR = os.path.join(DATA_DIR, 'DAVIS')
ANNOTATIONS_DIR = os.path.join(DAVIS_DIR, 'Annotations', '480p')
IMAGES_DIR = os.path.join(DAVIS_DIR, 'JPEGImages', '480p')

def get_annotation(image_set):
  from itertools import product

  img = cv2.imread(os.path.join(ANNOTATIONS_DIR, image_set, '00000.png'))[:,:,::-1]
  mask = np.zeros(img.shape[:2], np.int32)
  color_dict = {0:0}
  colors = [np.array([0,0,0], np.uint8)]
  count = 1
  for i, j in product(range(img.shape[0]), range(img.shape[1])):
    pixel = img[i,j]
    _pixel = pixel[0] | pixel[1] << 8 | pixel[2] << 16
    if _pixel in color_dict:
      mask[i,j] = color_dict[_pixel]
    else:
      mask[i,j] = count
      color_dict[_pixel] = count
      colors.append(pixel)
      count += 1
  return mask, np.array(colors)

def get_images(image_set):
  img_dir = os.path.join(IMAGES_DIR, image_set)
  img_files = os.listdir(img_dir)
  img_files.sort()
  return np.array([cv2.imread(os.path.join(img_dir, f))[:,:,::-1] for f in img_files])

def apply_mask(image, mask):
  return mask[:,:,0:1] * image + (mask[:,:,1:,np.newaxis] * colors[1:].reshape(1,1,-1,3)).sum(2)

def main():
  if not os.path.exists(DAVIS_DIR):
    from dataset import download_davis, _davis_url
    download_davis(_davis_url, DATA_DIR)
      
  colorizer = Colorizer(create_feature_extractor())
  colorizer.load_weights(os.path.join(MODEL_DIR, 'colorizer'))
    
  image_sets_file = os.path.join(DAVIS_DIR, 'ImageSets', '2017', 'test-dev.txt')
  with open(image_sets_file) as f:
    image_sets = list(map(lambda x: x.strip(), f.readlines()))
  image_set = np.random.choice(image_sets)

  annotation_large, colors = get_annotation(image_set)
  colors = colors / 255.
  n_colors = colors.shape[0]
  annotation_large = tf.one_hot(annotation_large, n_colors).numpy()
  annotation = cv2.resize(annotation_large, (32,32))

  all_images = get_images(image_set) / 255.
  all_images_gray = all_images.mean(3) * 2. - 1.
  all_images_gray = np.array([cv2.resize(img, (256,256)) for img in all_images_gray])
  all_images_gray = all_images_gray[...,np.newaxis]#.repeat(3, -1)

  plt.subplot(3,1,1)
  plt.imshow(apply_mask(all_images[0], annotation_large))
  plt.axis('off')

  annotations = annotation[np.newaxis].repeat(3, 0)
  for t in range(1, 3):
    images = all_images_gray[[max(t-3,0),max(t-2,0),t-1,t]]
    #print(images[np.newaxis,:3].shape)
    predictions = colorizer({
        'reference_images': np.float32(images[np.newaxis,:3]),
        'target_images': np.float32(images[np.newaxis,3:]),
        'reference_labels': np.float32(annotations[np.newaxis]),
        'temperature': 0.5})
    annotations = np.roll(annotations, -1, 0)
    annotations[-1] = predictions[0,0].numpy()
    mask = cv2.resize(predictions[0,0].numpy(), (all_images.shape[2], all_images.shape[1]))
    plt.subplot(3,1,t+1)
    plt.imshow(apply_mask(all_images[t], mask))
    plt.axis('off')
  plt.show()
  
if __name__=='__main__':
  main()
