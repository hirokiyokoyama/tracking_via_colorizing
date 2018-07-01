import tensorflow as tf
import numpy as np
import os
from nets import feature_extractor_resnet, colorizer
from dataset import create_ref_target_generator
from clustering import lab_to_labels, labels_to_lab
from clustering import num_clusters

NUM_REF = 3
NUM_TARGET = 1
FEATURE_DIM = 256
LEARNING_RATE = 0.001
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'data', 'model')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

##### create dataset
data = create_ref_target_generator(NUM_REF, NUM_TARGET).repeat().map(lambda x,y: tf.concat([x,y], 0))
image_gen = data.map(lambda x: tf.image.resize_images(x, [256,256]))
label_gen = data.map(lambda x: tf.image.resize_images(x, [32,32])).map(lab_to_labels)

# Lab image, [N,256,256,3], can be fed at sess.run
images = image_gen.make_one_shot_iterator().get_next(name='images')
# color labels (or other categorical data), [N,32,32,d], can be fed at sess.run
labels = label_gen.make_one_shot_iterator().get_next(name='labels')

##### extract features from gray scale image (only L channel) using CNN
feature_map = feature_extractor_resnet(images[:,:,:,0:1], dim=FEATURE_DIM)
# rename with tf.identity so that it can be easily fetched/fed at sess.run
feature_map = tf.identity(feature_map, name='features')

##### predict the color (or other category) on the basis of the features
end_points = colorizer(feature_map[:NUM_REF], tf.one_hot(labels[:NUM_REF], num_clusters),
                       feature_map[NUM_REF:], labels[NUM_REF:])
prediction = tf.identity(end_points['predictions'], name='predictions')
prediction_lab = labels_to_lab(prediction)
loss = tf.reduce_mean(end_points['losses'])
train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

##### summaries
loss_summary = tf.summary.scalar('loss', loss)
ph_target_img = tf.placeholder(tf.float32, shape=[1,None,None,3])
ph_vis_pred = tf.placeholder(tf.float32, shape=[1,None,None,3])
ph_vis_feat = tf.placeholder(tf.float32, shape=[1,None,None,3])
image_summary = tf.summary.merge([tf.summary.image('target_image', ph_target_img),
                                  tf.summary.image('visualized_prediction', ph_vis_pred),
                                  tf.summary.image('visualized_feature', ph_vis_feat)])
writer = tf.summary.FileWriter(MODEL_DIR)

##### create session and initialize
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

##### main loop
import cv2
from sklearn.decomposition import PCA
pca = PCA(n_components=3)

for i in xrange(100000):
    if i % 100 != 0:
        _, summary = sess.run([train_op, loss_summary])
        # summarize only loss
        writer.add_summary(summary, i)
    else:
        img, feat, pred, _, summary = sess.run([images,
                                                feature_map,
                                                prediction_lab,
                                                train_op,
                                                loss_summary])
        # summarize loss
        writer.add_summary(summary, i)

        # and images (doing some stuff to visualize outside the tf session)
        target_img = cv2.cvtColor(img[NUM_REF], cv2.COLOR_LAB2RGB)
        vis_pred = np.dstack([img[NUM_REF,:,:,0:1], cv2.resize(pred[0,:,:,1:], img.shape[1:3])])
        vis_pred = cv2.cvtColor(vis_pred, cv2.COLOR_LAB2RGB)
        feat_flat = feat[NUM_REF].reshape(-1, feat.shape[-1])
        pca.fit(feat_flat)
        feat_flat = pca.transform(feat_flat)
        feat_flat /= np.abs(feat_flat).max()
        feat_flat = (feat_flat + 1) / 2
        vis_feat = feat_flat.reshape(feat.shape[1:3]+(3,))
        summary = sess.run(image_summary, {ph_target_img: [target_img],
                                           ph_vis_pred: [vis_pred],
                                           ph_vis_feat: [vis_feat]})
        writer.add_summary(summary, i)
        
    if i % 1000 == 0:
        # save the model
        saver.save(sess, os.path.join(MODEL_DIR, 'model.ckpt'), global_step=i)
