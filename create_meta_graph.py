import tensorflow as tf
import numpy as np
import os
from nets import feature_extractor_resnet as feature_extractor
from nets import colorizer
from clustering import Clustering, visualize_ab_clusters

global_step = tf.Variable(0, trainable=False)
for _ in range(4): tf.Variable(0, trainable=False, collections=['temp'])

# If OOM happens, try smaller FEATURE_DIM and/or BATCH_SIZE
NUM_REF = 3
NUM_TARGET = 1
NUM_CLUSTERS = 16
KMEANS_STEPS_PER_ITERATION = 100
FEATURE_DIM = 128
LEARNING_RATE = 0.00001
INITIAL_WEIGHT = 3.
BATCH_SIZE = 8
TRAIN_INTERVAL = 1
IMAGE_SIZE = [256,256]
FEATURE_MAP_SIZE = [32,32] # IMAGE_SIZE/8 (depends on CNN)
HISTORY_CAPACITY = 1000
MIN_HISTORY_SIZE = 300
WEIGHT_DECAY = 0.0001
BATCH_NORM_DECAY = 0.999
BATCH_RENORM_DECAY = 0.99
USE_CONV3D = False
_t = tf.cast(global_step, tf.float32)
BATCH_RENORM_RMAX = tf.train.piecewise_constant(
    global_step, [2000, 2000+35000], [1., (_t-2000.)*(2./35000.)+1., 3.]) # 1. -> 3.
BATCH_RENORM_DMAX = tf.train.piecewise_constant(
    global_step, [2000, 2000+20000], [0., (_t-2000.)*(5./20000.), 5.]) # 0. -> 5.
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'data', 'model')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

image_batch = tf.placeholder(
    tf.float32, shape = [None, NUM_REF+NUM_TARGET]+IMAGE_SIZE+[3], name='images')

##### color clustering
kmeans = Clustering(tf.reshape(image_batch[:,:,:,:,1:], [-1,2]), NUM_CLUSTERS,
                    mini_batch_steps_per_iteration=KMEANS_STEPS_PER_ITERATION)
image_batch_flat = tf.reshape(image_batch, [-1]+IMAGE_SIZE+[3])
labels = tf.image.resize_images(image_batch_flat, FEATURE_MAP_SIZE)
labels = kmeans.lab_to_labels(labels)
labels = tf.reshape(labels, [-1,NUM_REF+NUM_TARGET]+FEATURE_MAP_SIZE)
labels = tf.placeholder_with_default(
    labels, [None, NUM_REF+NUM_TARGET]+FEATURE_MAP_SIZE, name='labels')

##### extract features from gray scale image (only L channel) using CNN
if USE_CONV3D:
    inputs = image_batch[:,:,:,:,0:1]
else:
    inputs = image_batch_flat[:,:,:,0:1]
is_training = tf.placeholder_with_default(False, [], name='is_training')
feature_map = feature_extractor(inputs,
                                dim = FEATURE_DIM,
                                weight_decay = WEIGHT_DECAY,
                                batch_norm_decay = BATCH_NORM_DECAY,
                                batch_renorm_decay = BATCH_RENORM_DECAY,
                                batch_renorm_rmax = BATCH_RENORM_RMAX,
                                batch_renorm_dmax = BATCH_RENORM_DMAX,
                                is_training = is_training,
                                use_conv3d = USE_CONV3D)
if not USE_CONV3D:
    feature_map = tf.reshape(
        feature_map, [-1,NUM_REF+NUM_TARGET]+FEATURE_MAP_SIZE+[FEATURE_DIM])
# rename with tf.identity so that it can be easily fetched/fed at sess.run
feature_map = tf.identity(feature_map, name='features')

##### predict the color (or other category) on the basis of the features
def loop_body(i, losses, predictions, predictions_lab):
    f = feature_map[i]
    l = labels[i]
    end_points = colorizer(f[:NUM_REF], tf.one_hot(l[:NUM_REF], NUM_CLUSTERS),
                           f[NUM_REF:], l[NUM_REF:])
    mean_losses = tf.reduce_mean(tf.reduce_mean(end_points['losses'], 2), 1)
    losses = tf.concat([losses, tf.expand_dims(mean_losses, 0)], 0)
    pred = end_points['predictions']
    predictions = tf.concat([predictions, tf.expand_dims(pred, 0)], 0)
    predictions_lab = tf.concat([predictions_lab, tf.expand_dims(kmeans.labels_to_lab(pred), 0)], 0)
    return i+1, losses, predictions, predictions_lab
loop_cond = lambda i, _1, _2, _3: tf.less(i, BATCH_SIZE)
loop_vars = [tf.constant(0),
             tf.zeros([0,NUM_TARGET], dtype=tf.float32),
             tf.zeros([0,NUM_TARGET]+FEATURE_MAP_SIZE+[NUM_CLUSTERS]),
             tf.zeros([0,NUM_TARGET]+FEATURE_MAP_SIZE+[3])]
shape_invariants = [tf.TensorShape([]),
                    tf.TensorShape([None,NUM_TARGET]),
                    tf.TensorShape([None,NUM_TARGET]+FEATURE_MAP_SIZE+[NUM_CLUSTERS]),
                    tf.TensorShape([None,NUM_TARGET]+FEATURE_MAP_SIZE+[3])]
_, losses, predictions, predictions_lab = tf.while_loop(loop_cond, loop_body, loop_vars,
                                                        shape_invariants=shape_invariants)
predictions = tf.identity(predictions, name='predictions')
loss = tf.reduce_mean(losses)

##### summaries
loss_summary = tf.summary.scalar('loss', loss)
ph_target_img = tf.placeholder(tf.float32, shape=[None,None,None,3])
ph_vis_pred = tf.placeholder(tf.float32, shape=[None,None,None,3])
ph_vis_feat = tf.placeholder(tf.float32, shape=[None,None,None,3])
kmeans_summary = tf.summary.image('kmeans_clusters',
                                  tf.expand_dims(visualize_ab_clusters(kmeans.cluster_centers), 0))
image_summary = tf.summary.merge([tf.summary.image('target_image', ph_target_img),
                                  tf.summary.image('visualized_prediction', ph_vis_pred),
                                  tf.summary.image('visualized_feature', ph_vis_feat)])
writer = tf.summary.FileWriter(MODEL_DIR)

##### create session and initialize
print tf.get_collection('variables', scope='Variable')
tf.train.export_meta_graph(os.path.join(MODEL_DIR, 'model.meta'))
