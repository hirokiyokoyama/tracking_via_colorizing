#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import cv2

from clustering import ColorClustering
from nets import create_feature_extractor, Colorizer

#############################################################
# If OOM happens, try smaller BATCH_SIZE
NUM_REF = 3
NUM_TARGET = 1
NUM_CLUSTERS = 16
KMEANS_STEPS_PER_ITERATION = 10
VIDEO_SIZE = 256           # must be same as in download_dataset.py and recommended to be close to max(IMAGE_SIZE)
IMAGE_SIZE = [256,256]

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'model')

BATCH_SIZE = 4
#############################################################

def create_dataset(kinetics, video_dir,
                   size = [256, 256],
                   num_frames = 4):
  if size is None:
    size = [None, None]

  def gen():
    for key, entry in kinetics.items():
      file = os.path.join(video_dir, key+'.mp4')
      cap = cv2.VideoCapture(file)
      frames = []
      while True:
        ret, frame = cap.read()
        if not ret:
          break
        frames.append(frame[:,:,::-1])
      if frames:
        yield entry['annotations']['label'], np.stack(frames, axis=0)
  dataset = tf.data.Dataset.from_generator(gen, (tf.string, tf.uint8), ([], [None,None,None,3]))

  def map_fn(label, frames):
    if size[0] is not None:
      frames = tf.image.resize(frames, size)
    return {
        'label': label,
        'frame_dataset': tf.data.Dataset.from_tensor_slices(frames)
    }
  dataset = dataset.map(map_fn)

  map_fn = lambda x: x['frame_dataset'].batch(
      num_frames, drop_remainder=True)
  dataset = dataset.interleave(map_fn, cycle_length=8)
  return dataset

def train_clusters(clustering, dataset):
  def preprocess(x):
    x = tf.cast(x, tf.float32) / 255.
    x = tf.image.rgb_to_yuv(x)
    return x[...,1:]

  for x in dataset.shuffle(100).map(preprocess).take(10):
    clustering.train(x)
  
def train_colorizer(colorizer, clustering, dataset,
                    save_dir = None,
                    batch_size = 4,
                    num_ref = 3):
  x, = dataset.take(1)
  feat_size = colorizer.feature_extractor.compute_output_shape(x.shape).as_list()[1:3]
  num_clusters = clustering.num_clusters
  
  def preprocess(x):
    x = tf.cast(x, tf.float32) / 255.
    x_small = tf.image.resize(x, feat_size)
    x = tf.image.rgb_to_yuv(x)
    x_small = tf.image.rgb_to_yuv(x_small)
    brightness = x[...,:1]*2. - 1.
    color = x_small[...,1:]
    labels = clustering.colors_to_labels(color)
    return {
        'reference_images': brightness[:num_ref],
        'target_images': brightness[num_ref:],
        'reference_labels': tf.one_hot(labels[:num_ref], num_clusters)
    }, labels[num_ref:]

  if save_dir:
    save_path = os.path.join(save_dir, 'model')
    class Callback(tf.keras.callbacks.Callback):
      def on_train_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
          colorizer.save_weights(save_path)
    callbacks = Callback()
  else:
    callbacks = None

  dataset = dataset.map(preprocess).batch(batch_size)
  colorizer.fit(
      dataset.prefetch(tf.data.experimental.AUTOTUNE),
      callbacks=Callback())
  
def main():
  if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

  clustering = ColorClustering(
      NUM_CLUSTERS,
      kmeans_steps_per_iteration = KMEANS_STEPS_PER_ITERATION)
  feature_extractor = create_feature_extractor()
  colorizer = Colorizer(feature_extractor)

  from dataset import load_kinetics
  kinetics = load_kinetics(DATA_DIR, 'train', download=False)
  if kinetics is None:
    raise Exception('Could not find Kinetics dataset. Please run download_dataset.py first.')
  video_dir = os.path.join(DATA_DIR, 'videos_'+str(VIDEO_SIZE))
  if not os.path.exists(video_dir):
    raise Exception(f'Could not find videos. Please check if VIDEO_SIZE=={VIDEO_SIZE} in download_dataset.py.')
  dataset = create_dataset(
      kinetics, video_dir,
      size = IMAGE_SIZE,
      num_frames = NUM_REF + NUM_TARGET)

  train_clusters(clustering, dataset)

  colorizer.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy')
    
  train_colorizer(colorizer, clustering, dataset,
                  model_dir = MODEL_DIR,
                  batch_size = BATCH_SIZE,
                  num_ref = NUM_REF)
  
if __name__=='__main__':
  main()
