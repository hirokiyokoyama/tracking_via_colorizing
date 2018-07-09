import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def feature_extractor_resnet(images,
                             layer = 'resnet_v1_101/block2/unit_3/bottleneck_v1',
                             dim = 256,
                             weight_decay = 0.0001,
                             batch_norm_decay = 0.997,
                             is_training = True):
    from tensorflow.contrib.slim.python.slim.nets import resnet_v1
    
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay, batch_norm_decay=batch_norm_decay)):
        _, end_points = resnet_v1.resnet_v1_101(images, 1000, is_training=is_training)
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            net = end_points[layer]
            # you can add convolutional layers here

            # the last layer without activation function
            feature_map = slim.conv2d(net, dim, [1,1],
                                      activation_fn=None,
                                      normalizer_fn=None)
    return feature_map

def feature_extractor_resnet_conv3d(images,
                                    dim = 256,
                                    weight_decay = 0.0001,
                                    batch_norm_decay = 0.997,
                                    is_training = True):
    from tensorflow.contrib.slim.python.slim.nets import resnet_v2
    
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay, batch_norm_decay=batch_norm_decay)):
        blocks = [
            resnet_v2.resnet_v2_block('block1', base_depth=16, num_units=3, stride=2),
            resnet_v2.resnet_v2_block('block2', base_depth=32, num_units=4, stride=2),
            resnet_v2.resnet_v2_block('block3', base_depth=64, num_units=6, stride=2),
            resnet_v2.resnet_v2_block('block4', base_depth=128, num_units=3, stride=1)
        ]
        _, end_points = resnet_v2.resnet_v2(images, blocks,
                                            is_training=is_training,
                                            include_root_block=False)
    with slim.arg_scope([slim.conv3d], stride=1, padding='SAME',
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            net = tf.expand_dims(end_points['resnet_v2/block3'], 0)
            net = slim.conv3d(net, dim, [3,3,3])
            net = slim.conv3d(net, dim, [3,3,3])
            net = slim.conv3d(net, dim, [3,3,3])
            net = slim.conv3d(net, dim, [3,3,3])[0]

            # the last layer without activation function
            feature_map = slim.conv2d(net, dim, [1,1],
                                      activation_fn=None,
                                      normalizer_fn=None)
    return feature_map

def colorizer(ref_features, ref_labels, target_features, target_labels=None):
    # ref_features: [N_REF,H,W,D], feature map from reference frames
    # ref_labels: [N_REF,H,W,d], category probabilities or one-hot vectors from reference frames
    # target_features: [N_TARGET,H,W,D], feature map from target frames
    # target_labels: ([N_TARGET,H,W,d],float) or ([N_TARGET,H,W],int), ground truth category probabilities or indices
    # (D denotes feature dimension, d denotes number of categories)
    bhw = tf.shape(target_features)[:-1]
    dim = tf.shape(ref_features)[-1]
    ref_features = tf.reshape(ref_features, [-1,1,dim])
    target_features = tf.reshape(target_features, [1,-1,dim])
    inner = tf.reduce_sum(ref_features * target_features, -1)
    temperature = tf.placeholder_with_default(1., [], name='temperature')
    weight_mat = tf.nn.softmax(inner/temperature, 0)

    ref_labels = tf.convert_to_tensor(ref_labels)
    if ref_labels.dtype in [tf.float16, tf.float32, tf.float64]:
        ref_labels = tf.reshape(ref_labels, [-1,1,tf.shape(ref_labels)[-1]])
    else:
        raise ValueError('ref_labels must be one-hot or probabilities, not indices')
        #ref_labels = tf.one_hot(tf.reshape(ref_labels,[-1,1]), )
    prediction = tf.reduce_sum(tf.expand_dims(weight_mat, -1) * ref_labels, 0)
    prediction = tf.reshape(prediction, tf.concat([bhw, [-1]], 0))
    end_points = {'logit_matrix': inner,
                  'weight_matrix': weight_mat,
                  'temperature': temperature,
                  'predictions': prediction}
    
    if target_labels is not None:
        target_labels = tf.convert_to_tensor(target_labels)
        if target_labels.dtype in [tf.float16, tf.float32, tf.float64]:
            fn = tf.nn.softmax_cross_entropy_with_logits
        else:
            fn = tf.nn.sparse_softmax_cross_entropy_with_logits
        end_points['losses'] = fn(logits=prediction, labels=target_labels)
    return end_points
