import tensorflow as tf
import numpy as np
import os
slim = tf.contrib.slim

def estimator(ref_features, ref_labels, target_features, target_labels=None):
    bhw = tf.shape(target_features)[:-1]
    dim = tf.shape(ref_features)[-1]
    ref_features = tf.reshape(ref_features, [-1,1,dim])
    target_features = tf.reshape(target_features, [1,-1,dim])
    inner = tf.reduce_sum(ref_features * target_features, -1)
    weight_mat = tf.nn.softmax(inner, 1)

    ref_labels = tf.convert_to_tensor(ref_labels)
    if ref_labels.dtype in [tf.float16, tf.float32, tf.float64]:
        ref_labels = tf.reshape(ref_labels, [-1,1,tf.shape(ref_labels)[-1]])
    else:
        raise ValueError('ref_labels must be one-hot')
        #ref_labels = tf.one_hot(tf.reshape(ref_labels,[-1,1]), )
    prediction = tf.reduce_sum(tf.expand_dims(weight_mat, -1) * ref_labels, 0)
    prediction = tf.reshape(prediction, tf.concat([bhw, [-1]], 0))
    end_points = {'logit_matrix': inner, 'weight_matrix': weight_mat, 'predictions': prediction}
    
    if target_labels is not None:
        target_labels = tf.convert_to_tensor(target_labels)
        if target_labels.dtype in [tf.float16, tf.float32, tf.float64]:
            fn = tf.nn.softmax_cross_entropy_with_logits
        else:
            fn = tf.nn.sparse_softmax_cross_entropy_with_logits
        end_points['losses'] = fn(logits=prediction, labels=target_labels)
    return end_points

from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from dataset import create_ref_target_generator
from clustering import lab_to_labels, labels_to_lab
from clustering import num_clusters

NUM_REF = 3
NUM_TARGET = 1
model_dir = os.path.join(os.path.dirname(__file__), 'data', 'model')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
#YOLOv3 upsample upper layer and concat with lower layer
#LAYER = 'resnet_v1_101/block2/unit_3/bottleneck_v1/conv3'
LAYER = 'resnet_v1_101/block2/unit_3/bottleneck_v1'
DIM = 256

data = create_ref_target_generator(NUM_REF, NUM_TARGET).repeat().map(lambda x,y: tf.concat([x,y], 0))
image_gen = data.map(lambda x: tf.image.resize_images(x, [256,256])).make_one_shot_iterator()
label_gen = data.map(lambda x: tf.image.resize_images(x, [32,32])).map(lab_to_labels).make_one_shot_iterator()

images = image_gen.get_next(name='images') # Lab image, [N,256,256,3], can be fed at sess.run
labels = label_gen.get_next(name='labels') # color labels (or other categorical data), [N,32,32,d], can be fed at sess.run

# extract features from gray scale image (only L channel) using CNN
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    _, end_points = resnet_v1.resnet_v1_101(images[:,:,:,0:1], 1000, is_training=True)
with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
                    activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
  with slim.arg_scope([slim.batch_norm], is_training=True):
      net = end_points[LAYER]
      feature_map = slim.conv2d(net, DIM, [1,1],
                                activation_fn=None,
                                normalizer_fn=None)
feature_map = tf.identity(feature_map, name='features') # using tf.identity to rename so that it can be easily fetched/fed at sess.run

# predict the color (or other category) on the basis of the features
end_points = estimator(feature_map[:NUM_REF], tf.one_hot(labels[:NUM_REF], num_clusters),
                       feature_map[NUM_REF:], labels[NUM_REF:])
prediction = tf.identity(end_points['predictions'], name='predictions')
prediction_lab = labels_to_lab(prediction)
loss = tf.reduce_sum(end_points['losses'])
train_op = tf.train.AdamOptimizer().minimize(loss)

# summaries
loss_summary = tf.summary.scalar('loss', loss)
ph_target_img = tf.placeholder(tf.float32, shape=[1,None,None,3])
ph_vis_pred = tf.placeholder(tf.float32, shape=[1,None,None,3])
ph_vis_feat = tf.placeholder(tf.float32, shape=[1,None,None,3])
image_summary = tf.summary.merge([tf.summary.image('target_image', ph_target_img),
                                  tf.summary.image('visualized_prediction', ph_vis_pred),
                                  tf.summary.image('visualized_feature', ph_vis_feat)])

sess = tf.Session()
saver = tf.train.Saver()
writer = tf.summary.FileWriter(model_dir)
sess.run(tf.global_variables_initializer())

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
        vis_pred = cv2.cvtColor(pred[0], cv2.COLOR_LAB2RGB)
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
        saver.save(sess, os.path.join(model_dir, 'model.ckpt'), global_step=i)
