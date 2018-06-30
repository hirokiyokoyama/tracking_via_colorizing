import tensorflow as tf
import numpy as np

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
from clustering import lab_to_labels
from clustering import num_clusters

NUM_REF = 3
NUM_TARGET = 1
data = create_ref_target_generator(NUM_REF, NUM_TARGET).repeat().map(lambda x,y: tf.concat([x,y], 0))
image_gen = data.map(lambda x: tf.image.resize_images(x[:,:,:,0:1], [256,256])).make_one_shot_iterator()
label_gen = data.map(lambda x: tf.image.resize_images(x, [32,32])).map(lab_to_labels).make_one_shot_iterator()

#YOLOv3 upsample upper layer and concat with lower layer
LAYER = 'resnet_v1_101/block2/unit_3/bottleneck_v1/conv3'
DIM = 512

images = image_gen.get_next()
labels = label_gen.get_next()
#images = tf.placeholder(tf.float32, shape=[None,None,None,1])
#labels = tf.placeholder(tf.int32, shape=[None,None,None])
#one_hot = tf.one_hot(labels, num_clusters)
with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
    _, end_points = resnet_v1.resnet_v1_101(images, 1000, is_training=True)
feature_map = end_points[LAYER][:,:,:,:256]

end_points = estimator(feature_map[:NUM_REF], tf.one_hot(labels[:NUM_REF], num_clusters),
                       feature_map[NUM_REF:], labels[NUM_REF:])
loss = tf.reduce_sum(end_points['losses'])
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(train_op)
