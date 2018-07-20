import tensorflow as tf
import numpy as np
import os
from nets import feature_extractor_resnet_conv3d as feature_extractor
from nets import colorizer
from dataset import create_ref_target_generator
from clustering import Clustering, visualize_ab_clusters
from replay import PrioritizedHistory

global_step = tf.Variable(0, trainable=False)

# If OOM happens, try smaller FEATURE_DIM and/or BATCH_SIZE
NUM_REF = 3
NUM_TARGET = 1
NUM_CLUSTERS = 16
KMEANS_STEPS_PER_ITERATION = 20
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
_t = tf.cast(global_step, tf.float32)
BATCH_RENORM_RMAX = tf.train.piecewise_constant(global_step,
                                                [5000, 5000+35000],
                                                [1., (_t-5000.)*(2./35000.)+1., 3.]) # 1. -> 3.
BATCH_RENORM_DMAX = tf.train.piecewise_constant(global_step,
                                                [5000, 5000+20000],
                                                [0., (_t-5000.)*(5./20000.), 5.]) # 0. -> 5.
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'data', 'model')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

##### create dataset
data = create_ref_target_generator(NUM_REF, NUM_TARGET).repeat().map(lambda x,y: tf.concat([x,y], 0))

raw_images = data.make_one_shot_iterator().get_next()
images = tf.image.resize_images(raw_images, IMAGE_SIZE)

##### create history
history = PrioritizedHistory({'images': (images.get_shape().as_list(), tf.float32)},
                             capacity=HISTORY_CAPACITY,
                             device='/cpu:0')
append_op = history.append({'images': images}, INITIAL_WEIGHT)
batch_inds, batch_data = history.sample(BATCH_SIZE)
image_batch = tf.identity(batch_data['images'], name='image_batch') #[N,NUM_REF+NUM_TARGET,H,W,C]

##### color clustering
kmeans = Clustering(tf.reshape(image_batch[:,:,:,:,1:], [-1,2]), NUM_CLUSTERS,
                    mini_batch_steps_per_iteration=KMEANS_STEPS_PER_ITERATION)
image_batch_flat = tf.reshape(image_batch, [-1]+IMAGE_SIZE+[3])
labels = tf.image.resize_images(image_batch_flat, FEATURE_MAP_SIZE)
labels = kmeans.lab_to_labels(labels)
labels = tf.reshape(labels, [BATCH_SIZE,NUM_REF+NUM_TARGET]+FEATURE_MAP_SIZE,
                    name='labels')

##### extract features from gray scale image (only L channel) using CNN
is_training = tf.placeholder_with_default(False, [], name='is_training')
feature_map = feature_extractor(image_batch_flat[:,:,:,0:1],
                                dim = FEATURE_DIM,
                                weight_decay = WEIGHT_DECAY,
                                batch_norm_decay = BATCH_NORM_DECAY,
                                is_training = is_training)
# rename with tf.identity so that it can be easily fetched/fed at sess.run
feature_map = tf.reshape(feature_map, [BATCH_SIZE,NUM_REF+NUM_TARGET]+FEATURE_MAP_SIZE+[FEATURE_DIM],
                         name='features')

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

##### update history
update_weight_op = tf.while_loop(lambda i: tf.less(i, BATCH_SIZE),
                                 lambda i: history.update_weight(batch_inds[i], tf.reduce_mean(losses[i])),
                                 [tf.constant(0)])
tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_weight_op)

##### training
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)

##### summaries
loss_summary = tf.summary.scalar('loss', loss)
history_weights_summary = tf.summary.histogram('history_weights', history._weights)
loss_summary = tf.summary.merge([loss_summary, history_weights_summary])
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
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
latest_ckpt = tf.train.latest_checkpoint(MODEL_DIR)
sess.run(history.initializer)
if latest_ckpt is not None:
    print 'Restoring from %s.' % latest_ckpt
    saver.restore(sess, latest_ckpt)
    kmeans_initialized = True
else:
    print 'Starting with a new model.'
    kmeans_initialized = False

##### main loop
import cv2
from sklearn.decomposition import PCA
pca = PCA(n_components=3)

i = 0
while True:
    sess.run(append_op)
    i += 1
    
    if i >= MIN_HISTORY_SIZE and i % TRAIN_INTERVAL == 0:
        j = tf.train.global_step(sess, global_step)
        print 'Train step', j
        if not kmeans_initialized:
            sess.run(kmeans.init_op)
        
        if j % 10 == 0:
            _, summary = sess.run([kmeans.train_op, kmeans_summary])
            writer.add_summary(summary, j)
        if j % 100 != 0:
            _, summary = sess.run([train_op, loss_summary], {is_training: True})
            # summarize only loss
            writer.add_summary(summary, j)
        else:
            imgs, feats, preds, _, summary = sess.run([image_batch,
                                                       feature_map,
                                                       predictions_lab,
                                                       train_op,
                                                       loss_summary], {is_training: True})
            # summarize loss
            writer.add_summary(summary, j)

            # and images (doing some stuff to visualize outside the tf session)
            target_img = np.zeros([BATCH_SIZE]+IMAGE_SIZE+[3])
            vis_pred = np.zeros([BATCH_SIZE]+IMAGE_SIZE+[3])
            vis_feat = np.zeros([BATCH_SIZE]+FEATURE_MAP_SIZE+[3])
            for k in xrange(BATCH_SIZE):
                img = imgs[k,NUM_REF]
                pred = preds[k,0]
                
                target_img[k] = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
                
                pred = np.dstack([img[:,:,0:1], cv2.resize(pred[:,:,1:], tuple(IMAGE_SIZE[::-1]))])
                vis_pred[k] = cv2.cvtColor(pred, cv2.COLOR_LAB2RGB)
                
                feat_flat = feats[k, NUM_REF].reshape(-1, FEATURE_DIM)
                pca.fit(feat_flat)
                feat_flat = pca.transform(feat_flat)
                feat_flat /= np.abs(feat_flat).max()
                feat_flat = (feat_flat + 1) / 2
                vis_feat[k] = feat_flat.reshape(FEATURE_MAP_SIZE+[3])
            summary = sess.run(image_summary, {ph_target_img: target_img,
                                               ph_vis_pred: vis_pred,
                                               ph_vis_feat: vis_feat})
            writer.add_summary(summary, j)
        
        if (j+1) % 1000 == 0:
            # save the model
            saver.save(sess, os.path.join(MODEL_DIR, 'model.ckpt'), global_step=global_step)
