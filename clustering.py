import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.factorization import KMeans, KMeansClustering

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

# creates operations in the default graph
class Clustering:
    def __init__(self, inputs, num_clusters, mini_batch_steps_per_iteration=100):
        self.num_clusters = tf.convert_to_tensor(num_clusters)
        self.kmeans = KMeans(inputs, self.num_clusters,
                             use_mini_batch=True,
                             mini_batch_steps_per_iteration=mini_batch_steps_per_iteration)
        out = self.kmeans.training_graph()
        self.cluster_centers = tf.get_default_graph().get_tensor_by_name('clusters:0')
        self.all_scores = out[0][0]
        self.cluster_index = out[1][0]
        self.scores = out[2][0]
        self.cluster_centers_initialized = out[3]
        self.init_op = out[4]
        self.train_op = out[5]

    def lab_to_labels(self, images, name='lab_to_labels'):
        a = tf.reshape(self.cluster_centers[:,0], [1,1,1,-1])
        b = tf.reshape(self.cluster_centers[:,1], [1,1,1,-1])
        da = tf.expand_dims(images[:,:,:,1],3) - a
        db = tf.expand_dims(images[:,:,:,2],3) - b
        d = tf.square(da) + tf.square(db)
        return tf.argmin(d, 3, name=name)

    def labels_to_lab(self, labels, name='labels_to_lab'):
        if labels.dtype in [tf.float16, tf.float32, tf.float64]:
            l = tf.cast(tf.expand_dims(labels,-1), tf.float32)
            c = tf.reshape(self.cluster_centers, [1,1,1,-1,2])
            ab = tf.reduce_sum(l * c, 3)
        else:
            ab = tf.gather(self.cluster_centers, labels)
        l = tf.ones(tf.shape(ab)[:-1], tf.float32) * 75
        return tf.concat([tf.expand_dims(l,-1), ab], 3, name=name)
    
# TODO: Now only labels_to_lab and lab_to_labels are tensorflow operation.
# Make others as well using tensorflow.contrib.factorization.KMeans
class Clustering_old:
    def __init__(self, model_dir, num_clusters):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), 'data', 'kmeans')
        if num_clusters is None:
            num_clusters = 16
        self.num_clusters = num_clusters
        self.kmeans = KMeansClustering(num_clusters=num_clusters,
                                       distance_metric=KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE,
                                       mini_batch_steps_per_iteration = 100,
                                       use_mini_batch=True,
                                       model_dir=model_dir)

    def lab_to_labels(self, image_batch):
        cluster_centers = tf.py_func(self.kmeans.cluster_centers, [], tf.float32)
        a = tf.reshape(cluster_centers[:,0], [1,1,1,-1])
        b = tf.reshape(cluster_centers[:,1], [1,1,1,-1])
        da = tf.expand_dims(image_batch[:,:,:,1],3) - a
        db = tf.expand_dims(image_batch[:,:,:,2],3) - b
        d = tf.square(da) + tf.square(db)
        return tf.argmin(d, 3)

    def labels_to_lab(self, labels):
        cluster_centers = tf.py_func(self.kmeans.cluster_centers, [], tf.float32)
        if labels.dtype in [np.float16, np.float32, np.float64]:
            l = tf.cast(tf.expand_dims(labels,-1), tf.float32)
            c = tf.reshape(cluster_centers, [1,1,1,-1,2])
            ab = tf.reduce_sum(l * c, 3)
        else:
            ab = tf.gather(cluster_centers, labels)
        l = tf.ones(tf.shape(ab)[:-1], tf.float32) * 75
        return tf.concat([tf.expand_dims(l,-1), ab], 3)

    def train(self, img_batch):
        input_fn = tf.train.limit_epochs(
            tf.convert_to_tensor(img_batch, dtype=tf.float32), num_epochs=1)
        return self.kmeans.train(input_fn)

    def cluster_centers(self):
        return self.kmeans.cluster_centers()

    def visualize(self):
        import cv2
        sess = tf.Session()
        labels = np.arange(16).reshape([1,1,16])
        lab_img = sess.run(self.labels_to_lab(labels))[0]
        bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
        bgr_img = np.repeat(np.repeat(bgr_img, 100, 0), 30, 1)
        cv2.imshow('Color palette', bgr_img)
        cv2.waitKey(0)

if __name__=='__main__':
    from dataset import create_batch_generator

    kmeans = Clustering_old()
    def preprocess(x):
        x = tf.image.resize_images(x, [32,32])
        return tf.reshape(x[:,:,:,1:], [-1,2])
    kmeans.kmeans.train(lambda: create_batch_generator(1).map(preprocess))
    print kmeans.cluster_centers()
