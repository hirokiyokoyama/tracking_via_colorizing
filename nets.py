import tensorflow as tf
import numpy as np

def copy_labels(ref_features, ref_labels, target_features, temperature=1.):
  # ref_features: [N_REF,H,W,C], feature map from reference frames
  # ref_labels: [N_REF,H,W,D], category probabilities or one-hot vectors from reference frames
  # target_features: [N_TARGET,H,W,C], feature map from target frames
  # target_labels: ([N_TARGET,H,W,D],float) or ([N_TARGET,H,W],int), ground truth category probabilities or indices
  # (C denotes feature dimension, D denotes number of categories)

  ref_labels = tf.convert_to_tensor(ref_labels)
  if ref_labels.dtype in [tf.float16, tf.float32, tf.float64]:
    ref_labels = tf.reshape(ref_labels, [-1,1,tf.shape(ref_labels)[-1]])
  else:
    raise ValueError('ref_labels must be one-hot or probabilities, not indices')

  bhw = tf.shape(target_features)[:-1]
  dim = tf.shape(ref_features)[-1]

  #ref_features = tf.reshape(ref_features, [-1,1,dim])
  #target_features = tf.reshape(target_features, [1,-1,dim])
  #inner = tf.reduce_sum(ref_features * target_features, axis=-1)
  ref_features = tf.reshape(ref_features, [-1,dim])
  target_features = tf.reshape(target_features, [-1,dim])
  inner = tf.matmul(ref_features, target_features, transpose_b=True)

  weight_mat = tf.nn.softmax(inner/temperature, axis=0)

  prediction = tf.reduce_sum(tf.expand_dims(weight_mat, -1) * ref_labels, 0)
  prediction = tf.reshape(prediction, tf.concat([bhw, [-1]], 0))
  return prediction

class Colorizer(tf.keras.Model):
  def __init__(self, feature_extractor):
    super().__init__()
    self.feature_extractor = feature_extractor

  def call(self, inputs, training=False):
    #ref_images = tf.keras.Input([num_ref, None, None, 1])
    #target_images = tf.keras.Input([num_target, None, None, 1])
    #ref_labels = tf.keras.Input([num_ref,None,None,None])
    ref_images = inputs['reference_images']
    target_images = inputs['target_images']
    ref_labels = inputs['reference_labels']
    temperature = inputs.get('temperature', 1.)

    num_ref = tf.shape(ref_images)[1]
    num_target = tf.shape(target_images)[1]

    inputs = tf.concat([ref_images, target_images], axis=1)
    N,T,H,W,C = tf.unstack(tf.shape(inputs))
    inputs_flat = tf.reshape(inputs, [N*T,H,W,C])

    features_flat = self.feature_extractor(inputs_flat, training=training)
    _,h,w,c = tf.unstack(tf.shape(features_flat))
    features = tf.reshape(features_flat, [N,T,h,w,c])
    ref_feat, target_feat = tf.split(features, [num_ref, num_target], axis=1)

    arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in tf.range(N):
      prediction = copy_labels(ref_feat[i], ref_labels[i], target_feat[i], temperature)
      arr = arr.write(i, prediction)
    prediction = arr.stack()
    return prediction

def create_feature_extractor():
  from tensorflow.python.keras.applications import resnet

  x = tf.keras.Input([None, None, 1])
  y = x
  y = tf.keras.layers.Conv2D(16, 3, padding='SAME')(y)
  y = resnet.stack2(y, 16, 3, stride1=2, name='stack1')
  y = resnet.stack2(y, 32, 4, stride1=2, name='stack2')
  y = resnet.stack2(y, 64, 6, stride1=2, name='stack3')
  y = resnet.stack2(y, 128, 3, stride1=1, name='stack4')

  #y = tf.keras.layers.BatchNormalization()(y)
  #y = tf.keras.layers.ReLU()(y)
  #y = tf.keras.layers.Conv2D(, 3, )
  return tf.keras.Model(x, y)
