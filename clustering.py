import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.factorization import KMeansClustering

# TODO: Now only labels_to_lab and lab_to_labels are tensorflow operation.
# Make others as well using tensorflow.contrib.factorization.KMeans
class Clustering:
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

    kmeans = Clustering()
    def preprocess(x):
        x = tf.image.resize_images(x, [32,32])
        return tf.reshape(x[:,:,:,1:], [-1,2])
    kmeans.kmeans.train(lambda: create_batch_generator(1).map(preprocess))
    print kmeans.cluster_centers()
