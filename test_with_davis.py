#!/usr/bin/env python3

import tensorflow as tf
import cv2
import numpy as np
import os
from nets import create_feature_extractor, Colorizer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'model')

def apply_mask(images, masks, colors):
  colors = tf.reshape(colors[1:], [1,1,1,-1,3])
  bg = masks[...,0:1] * images
  fg = masks[...,1:,tf.newaxis] * colors
  fg = tf.reduce_sum(fg, axis=3)
  return bg + fg

class Davis:
  def __init__(self, base_dir):
    self.davis_dir = os.path.join(base_dir, 'DAVIS')
    if not os.path.exists(self.davis_dir):
      from dataset import download_davis, _davis_url
      download_davis(_davis_url, base_dir)

  @property
  def image_sets(self):
    image_sets_file = os.path.join(
        self.davis_dir, 'ImageSets', '2017', 'test-dev.txt')
    with open(image_sets_file) as f:
      image_sets = list(map(lambda x: x.strip(), f.readlines()))
    return image_sets

  def get_images(self, image_set):
    img_dir = os.path.join(
        self.davis_dir, 'JPEGImages', '480p', image_set)

    img_files = os.listdir(img_dir)
    img_files.sort()
    images = [cv2.imread(os.path.join(img_dir, f))[:,:,::-1] \
              for f in img_files]
    return np.array(images)

  def get_annotation(self, image_set):
    from itertools import product
    annotations_dir = os.path.join(
        self.davis_dir, 'Annotations', '480p')

    img = cv2.imread(os.path.join(annotations_dir, image_set, '00000.png'))[:,:,::-1]
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

def track_regions(colorizer, images, labels,
                  temperature = 0.5):
  images_gray = tf.reduce_mean(images, axis=3, keepdims=True) * 2. - 1.
  images_gray = tf.image.resize(images_gray, [256,256])
  n_images, orig_h, orig_w, c = tf.unstack(tf.shape(images))

  out_shape = colorizer.feature_extractor.compute_output_shape(
      images_gray.shape)
  labels_downsampled = tf.image.resize(labels, out_shape[1:3])

  labels_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  labels_array = labels_array.write(0, labels_downsampled)

  def loop_body(i, images, labels):
    begin = tf.maximum(i-3, 0)
    ref_images = images[begin:i]
    target_images = images[i:i+1]
    ref_labels = labels.gather(tf.range(begin, i))
    target_labels = colorizer({
        'reference_images': ref_images[tf.newaxis],
        'target_images': target_images[tf.newaxis],
        'reference_labels': ref_labels[tf.newaxis],
        'temperature': temperature})
    labels = labels.write(i, target_labels[0,0])
    return i+1, images, labels
  loop_cond = lambda i, _, labels: i < n_images
  loop_vars = tf.constant(1, dtype=tf.int32), images_gray, labels_array
  i, _, labels_array = tf.while_loop(loop_cond, loop_body, loop_vars)
  
  return tf.image.resize(labels_array.stack(), [orig_h, orig_w])

def make_animation(frames):
  plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
  patch = plt.imshow(frames[0])
  plt.axis('off')

  def animate(i):
    patch.set_data(frames[i])

  anim = FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=1000/30.0)
  return anim
    
def main(data_dir=DATA_DIR, model_dir=MODEL_DIR):
  davis = Davis(base_dir=data_dir)
  
  colorizer = Colorizer(create_feature_extractor())
  ckpt = tf.train.latest_checkpoint(MODEL_DIR)
  if ckpt:
    colorizer.load_weights(ckpt)

  image_set = np.random.choice(davis.image_sets)

  annotations, colors = davis.get_annotation(image_set)
  colors = tf.constant(colors / 255., dtype=tf.float32)
  n_colors = colors.shape[0]
  initial_labels = tf.one_hot(annotations, n_colors)

  images = tf.constant(davis.get_images(image_set) / 255.,
                       dtype = tf.float32)

  labels = track_regions(colorizer, images, initial_labels,
                         temperature = 0.5)
  results = apply_mask(images, labels, colors)
  make_animation(results)
  plt.show()
  
if __name__=='__main__':
  main(data_dir=DATA_DIR, model_dir=MODEL_DIR)
