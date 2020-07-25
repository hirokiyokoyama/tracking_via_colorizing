import tensorflow as tf
import numpy as np

class ColorClustering(object):
  def __init__(self, num_clusters,
               kmeans_steps_per_iteration = 10,
               model_dir = None):
    self._kmeans = tf.compat.v1.estimator.experimental.KMeans(
        num_clusters=num_clusters,
        use_mini_batch=True,
        mini_batch_steps_per_iteration=kmeans_steps_per_iteration,
        model_dir=model_dir)
    self.num_clusters = num_clusters

  def train(self, colors):
    dimension = tf.shape(colors)[-1]
    colors = tf.reshape(colors, [-1, dimension])

    def input_fn():
      #return tf.data.Dataset.from_tensors(colors.numpy())
      return tf.data.Dataset.from_tensor_slices(colors.numpy()).shuffle(10000).batch(100)
    self._kmeans.train(input_fn)

  def colors_to_labels(self, colors):
    cluster_centers = self.cluster_centers
    num_clusters, dimension = tf.unstack(tf.shape(cluster_centers))

    shape = tf.concat([[num_clusters], tf.ones([tf.rank(colors)-1], dtype=tf.int32), [dimension]], axis=0)
    cluster_centers = tf.reshape(cluster_centers, shape)
    colors = colors[tf.newaxis]

    d = tf.reduce_sum(tf.square(colors - cluster_centers), axis=-1)
    return tf.argmin(d, axis=0)

  def labels_to_colors(self, labels):
    if labels.dtype in [tf.float16, tf.float32, tf.float64]:
      cluster_centers = self.cluster_centers

      shape = tf.concat([tf.ones([tf.rank(labels)-1], dtype=tf.int32), tf.shape(cluster_centers)], axis=0)
      cluster_centers = tf.reshape(cluster_centers, shape)
      labels = labels[...,tf.newaxis]

      colors = tf.reduce_sum(labels * cluster_centers, -2)
    else:
      colors = tf.gather(self.cluster_centers, labels)
    return colors

  @property
  def cluster_centers(self):
    return self._kmeans.cluster_centers()

# a: left to right, b: top to bottom
__ab_space_image = None
def _ab_space_image():
    global __ab_space_image
    if __ab_space_image is not None:
        return __ab_space_image
    
    import cv2
    a = np.arange(-110,110)
    b = np.arange(-110,110)
    a, b = map(np.float32, np.meshgrid(a, b))
    l = np.ones(a.shape[:2], dtype=np.float32) * 50.
    lab = np.dstack((l,a,b))
    __ab_space_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return tf.convert_to_tensor(__ab_space_image)

def visualize_ab_clusters(clusters):
    image = _ab_space_image()
    center_indices = tf.cast(clusters[:,::-1] + tf.constant([[110, 110]], dtype=tf.float32), tf.int32)
    indices = []
    for i in range(-1,2):
        for j in range(-1,2):
            indices.append(center_indices + tf.constant([[i,j]]))
    indices = tf.concat(indices, 0)
    indices = tf.clip_by_value(indices, 0, tf.reduce_min(tf.shape(image)[0:2])-1)
    updates = tf.ones([tf.shape(indices)[0], 3], dtype=tf.int32)
    cond = tf.scatter_nd(indices, updates, tf.shape(image))
    vis = tf.where(tf.cast(cond, tf.bool), tf.zeros_like(image), image)
    return vis[::-1,:,:]
