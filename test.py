import tensorflow as tf
import numpy as np
import os
import cv2
from dataset import create_ref_target_generator
from clustering import lab_to_labels, labels_to_lab
from clustering import num_clusters

NUM_REF = 3
NUM_TARGET = 1
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'model', 'model.ckpt')

# load images
with tf.Graph().as_default() as graph:
    data = create_ref_target_generator(NUM_REF, NUM_TARGET).repeat().map(lambda x,y: tf.concat([x,y], 0))
    images = data.map(lambda x: tf.image.resize_images(x, [256,256])).make_one_shot_iterator().get_next()
    labels = data.map(lambda x: tf.image.resize_images(x, [32,32])).map(lab_to_labels).make_one_shot_iterator().get_next()
with tf.Session(graph=graph) as sess:
    _images, _labels = sess.run([images, labels])

with tf.Graph().as_default() as graph:
    saver = tf.train.import_meta_graph(MODEL_PATH+'-0.meta')

    # Lab image, [N,256,256,3]
    images = graph.get_tensor_by_name('images:0')
    # color labels (or other categorical data)
    labels = graph.get_tensor_by_name('labels:0')
    features = graph.get_tensor_by_name('features:0')
    predictions = graph.get_tensor_by_name('predictions:0')

with tf.Session(graph=graph) as sess:
    saver.restore(sess, MODEL_PATH)

    # fetch features and predictions separately for the sake of demonstration
    # features need only images
    _features = sess.run(features, {images: _images})
    print _feature.shape
    # predictions need images and labels
    _predictions = sess.run(predictions, {images: _images, labels: _labels})
    print _predictions.shape
