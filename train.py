import tensorflow as tf
import numpy as np
import os
from nets import feature_extractor_resnet as feature_extractor
from nets import colorizer
from dataset import create_ref_target_generator
from clustering import Clustering, visualize_ab_clusters

global_step = tf.Variable(0, trainable=False)

NUM_REF = 3
NUM_TARGET = 1
NUM_CLUSTERS = 16
KMEANS_STEPS_PER_ITERATION = 100
FEATURE_DIM = 128
LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.0001
BATCH_NORM_DECAY = 0.999
BATCH_RENORM_DECAY = 0.99
IMAGE_SIZE = [256,256]
FEATURE_MAP_SIZE = [32,32] # IMAGE_SIZE/8 (depends on CNN)
USE_CONV3D = False
_t = tf.cast(global_step, tf.float32)
BATCH_RENORM_RMAX = tf.train.piecewise_constant(
    global_step, [2000, 2000+35000], [1., (_t-5000.)*(2./35000.)+1., 3.]) # 1. -> 3.
BATCH_RENORM_DMAX = tf.train.piecewise_constant(
    global_step, [2000, 2000+20000], [0., (_t-5000.)*(5./20000.), 5.]) # 0. -> 5.
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'data', 'model')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

##### create dataset
data = create_ref_target_generator(NUM_REF, NUM_TARGET).repeat().map(lambda x,y: tf.concat([x,y], 0))

raw_images = data.make_one_shot_iterator().get_next()
# Lab image, [N,256,256,3], can be fed at sess.run
images = tf.image.resize_images(raw_images, IMAGE_SIZE)
image_batch = tf.expand_dims(images, 0)
image_batch = tf.placeholder_with_default(
    batch_data['images'], [None, NUM_REF+NUM_TARGET]+IMAGE_SIZE+[3], name='images')

##### color clustering
kmeans = Clustering(tf.reshape(image_batch[:,:,:,:,1:], [-1,2]), NUM_CLUSTERS,
                    mini_batch_steps_per_iteration=KMEANS_STEPS_PER_ITERATION)
image_batch_flat = tf.reshape(image_batch, [-1]+IMAGE_SIZE+[3])
# color labels (or other categorical data), [N,32,32,d], can be fed at sess.run
labels = tf.image.resize_images(image_batch_flat, FEATURE_MAP_SIZE)
labels = kmeans.lab_to_labels(labels)
labels = tf.reshape(labels, [1,NUM_REF+NUM_TARGET]+FEATURE_MAP_SIZE)
labels = tf.placeholder_with_default(
    labels, [None, NUM_REF+NUM_TARGET]+FEATURE_MAP_SIZE, name='labels')

##### extract features from gray scale image (only L channel) using CNN
if USE_CONV3D:
    inputs = image_batch[:,:,:,:,0:1]
else:
    inputs = image_batch_flat[:,:,:,0:1]
# can be fed at sess.run, False by default
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
        feature_map, [1,NUM_REF+NUM_TARGET]+FEATURE_MAP_SIZE+[FEATURE_DIM])
# rename with tf.identity so that it can be easily fetched/fed at sess.run
feature_map = tf.identity(feature_map, name='features')

##### predict the color (or other category) on the basis of the features
end_points = colorizer(feature_map[0,:NUM_REF], tf.one_hot(labels[0,:NUM_REF], NUM_CLUSTERS),
                       feature_map[0,NUM_REF:], labels[0,NUM_REF:])
prediction = end_points['predictions']
tf.expand_dims(prediction, 0, name='predictions')
prediction_lab = kmeans.labels_to_lab(prediction)
loss = tf.reduce_mean(end_points['losses'])

##### training
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)

##### summaries
loss_summary = tf.summary.scalar('loss', loss)
ph_target_img = tf.placeholder(tf.float32, shape=[1,None,None,3])
ph_vis_pred = tf.placeholder(tf.float32, shape=[1,None,None,3])
ph_vis_feat = tf.placeholder(tf.float32, shape=[1,None,None,3])
kmeans_summary = tf.summary.image('kmeans_clusters',
                                  tf.expand_dims(visualize_ab_clusters(kmeans.cluster_centers), 0))
image_summary = tf.summary.merge([tf.summary.image('target_image', ph_target_img),
                                  tf.summary.image('visualized_prediction', ph_vis_pred),
                                  tf.summary.image('visualized_feature', ph_vis_feat)])
writer = tf.summary.FileWriter(MODEL_DIR)

##### create session and initialize
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
latest_ckpt = tf.train.latest_checkpoint(MODEL_DIR)
if latest_ckpt is not None:
    saver.restore(sess, latest_ckpt)
else:
    sess.run(kmeans.init_op)

##### main loop
import cv2
from sklearn.decomposition import PCA
pca = PCA(n_components=3)

while True:
    i = tf.train.global_step(sess, global_step)
    if i % 10 == 0:
        _, summary = sess.run([kmeans.train_op, kmeans_summary])
        writer.add_summary(summary, i)
    if i % 100 != 0:
        _, summary = sess.run([train_op, loss_summary], {is_training: True})
        # summarize only loss
        writer.add_summary(summary, i)
    else:
        img, feat, pred, _, summary = sess.run([image_batch,
                                                feature_map,
                                                prediction_lab,
                                                train_op,
                                                loss_summary], {is_training: True})
        # summarize loss
        writer.add_summary(summary, i)

        # and images (doing some stuff to visualize outside the tf session)
        target_img = cv2.cvtColor(img[0,NUM_REF], cv2.COLOR_LAB2RGB)
        vis_pred = np.dstack([img[0,NUM_REF,:,:,0:1], cv2.resize(pred[0,:,:,1:], tuple(IMAGE_SIZE[::-1]))])
        vis_pred = cv2.cvtColor(vis_pred, cv2.COLOR_LAB2RGB)
        feat_flat = feat[0,NUM_REF].reshape(-1, FEATURE_DIM)
        pca.fit(feat_flat)
        feat_flat = pca.transform(feat_flat)
        feat_flat /= np.abs(feat_flat).max()
        feat_flat = (feat_flat + 1) / 2
        vis_feat = feat_flat.reshape(FEATURE_MAP_SIZE+[3])
        summary = sess.run(image_summary, {ph_target_img: [target_img],
                                           ph_vis_pred: [vis_pred],
                                           ph_vis_feat: [vis_feat]})
        writer.add_summary(summary, i)
        
    if (i+1) % 1000 == 0:
        # save the model
        saver.save(sess, os.path.join(MODEL_DIR, 'model.ckpt'), global_step=global_step)
