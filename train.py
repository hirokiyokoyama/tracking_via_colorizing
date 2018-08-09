import tensorflow as tf
import numpy as np
import os
from nets import feature_extractor_resnet as feature_extractor
from nets import colorizer
from dataset import create_ref_target_generator
from clustering import Clustering, visualize_ab_clusters
from replay import PrioritizedHistory

#############################################################
# If OOM happens, try smaller FEATURE_DIM and/or BATCH_SIZE
NUM_REF = 3
NUM_TARGET = 1
NUM_CLUSTERS = 16
KMEANS_STEPS_PER_ITERATION = 100
FEATURE_DIM = 128
LEARNING_RATE = lambda t: 0.00001
WEIGHT_DECAY = 0.0001
BATCH_NORM_DECAY = 0.999
BATCH_RENORM_DECAY = 0.99
IMAGE_SIZE = [256,256]
FEATURE_MAP_SIZE = [32,32] # IMAGE_SIZE/8 (depends on CNN)
USE_CONV3D = False
BATCH_RENORM_RMAX = lambda t: tf.train.piecewise_constant(
    t, [2000., 2000.+35000.], [1., (t-2000.)*(2./35000.)+1., 3.]) # 1. -> 3.
BATCH_RENORM_DMAX = lambda t: tf.train.piecewise_constant(
    t, [2000., 2000.+20000.], [0., (t-2000.)*(5./20000.), 5.]) # 0. -> 5.
LOSS_WEIGHTING_STRENGTH = 0.5
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'data', 'model')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

USE_HISTORY = True # if False, following constants are ignored
INITIAL_WEIGHT = 3.
BATCH_SIZE = 8
TRAIN_INTERVAL = 1
HISTORY_CAPACITY = 10000
MIN_HISTORY_SIZE = 3000
#############################################################

if not USE_HISTORY:
    TRAIN_INTERVAL = 1
    MIN_HISTORY_SIZE = 0
    BATCH_SIZE = 1

def kl_divergence(p, q, axis=-1):
    x = p*(tf.log(p)-tf.log(q))
    return tf.reduce_sum(tf.boolean_mask(x, tf.greater(p,0)), axis)

def _build_graph(image_batch):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    t = tf.cast(global_step, tf.float32)
    
    ##### color clustering
    kmeans = Clustering(tf.reshape(image_batch[:,:,:,:,1:], [-1,2]), NUM_CLUSTERS,
                        mini_batch_steps_per_iteration=KMEANS_STEPS_PER_ITERATION)
    image_batch_flat = tf.reshape(image_batch, [-1]+IMAGE_SIZE+[3])
    labels = tf.image.resize_images(image_batch_flat, FEATURE_MAP_SIZE)
    labels = kmeans.lab_to_labels(labels)
    labels = tf.reshape(labels, [-1,NUM_REF+NUM_TARGET]+FEATURE_MAP_SIZE, name='labels')

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
                                    batch_renorm_rmax = BATCH_RENORM_RMAX(t),
                                    batch_renorm_dmax = BATCH_RENORM_DMAX(t),
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
    predictions_lab = tf.identity(predictions_lab, name='predictions_lab')
    losses = tf.identity(losses, name='losses')

    ##### calculate differences between reference and target images
    #[BATCH_SIZE,NUM_REF+NUM_TARGET,NUM_CLUSTERS]
    pq = tf.reduce_mean(tf.reduce_mean(tf.one_hot(labels, NUM_CLUSTERS), 2), 2)
    q = tf.reduce_mean(pq[:,:NUM_REF,:], 1, keepdims=True)
    p = pq[:,NUM_REF:,:]
    #[BATCH_SIZE,NUM_TARGET]
    consistency = tf.identity(kl_divergence(p, q), name='consistency')
    loss_weights = tf.exp(-LOSS_WEIGHTING_STRENGTH*consistency, name='loss_weights')

    return kmeans

##### build a graph to export as a meta graph
with tf.Graph().as_default() as graph:
    image_batch = tf.placeholder(tf.float32,
                                 shape = [None, NUM_REF+NUM_TARGET]+IMAGE_SIZE+[3],
                                 name = 'images')
    _build_graph(image_batch)
    tf.train.export_meta_graph(os.path.join(MODEL_DIR, 'model.meta'))

##### create dataset
data = create_ref_target_generator(NUM_REF, NUM_TARGET).repeat().map(lambda x,y: tf.concat([x,y], 0))

raw_images = data.make_one_shot_iterator().get_next()
images = tf.image.resize_images(raw_images, IMAGE_SIZE)

##### create history
if USE_HISTORY:
    history = PrioritizedHistory({'images': (images.get_shape().as_list(), tf.float32)},
                                 capacity=HISTORY_CAPACITY,
                                 device='/cpu:0')
    append_op = history.append({'images': images}, INITIAL_WEIGHT)
    batch_inds, batch_data = history.sample(BATCH_SIZE)
    image_batch = tf.identity(batch_data['images'], name='images')
else:
    image_batch = tf.expand_dims(images, 0, name='images')

##### training
kmeans = _build_graph(image_batch)
graph = tf.get_default_graph()
global_step = graph.get_tensor_by_name('global_step:0')
t = tf.cast(global_step, tf.float32)
is_training = graph.get_tensor_by_name('is_training:0')
feature_map = graph.get_tensor_by_name('features:0')
predictions_lab = graph.get_tensor_by_name('predictions_lab:0')
losses = graph.get_tensor_by_name('losses:0')
consistency = graph.get_tensor_by_name('consistency:0')
loss_weights = graph.get_tensor_by_name('loss_weights:0')
loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(losses, -1), -1) * loss_weights)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE(t))\
    .minimize(loss, global_step = global_step)

##### update history
if USE_HISTORY:
    weights = tf.reduce_mean(tf.layers.flatten(losses), -1)
    with tf.control_dependencies([train_op]):
        train_op = history.update_weights(batch_inds, weights)

##### summaries
loss_summary = tf.summary.scalar('loss', loss)
if USE_HISTORY:
    history_weights_summary = tf.summary.histogram('history_weights', history._weights)
    loss_summary = tf.summary.merge([loss_summary, history_weights_summary])
loss_weights_summary = tf.summary.scalar('loss_weights', loss_weights)
loss_summary = tf.summary.merge([loss_summary, loss_weights_summary])
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
if USE_HISTORY:
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
    if USE_HISTORY:
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
            saver.save(sess, os.path.join(MODEL_DIR, 'model.ckpt'),
                       global_step = global_step,
                       write_meta_graph = False)
