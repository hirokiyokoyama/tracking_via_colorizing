import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.factorization import KMeansClustering
from dataset import create_batch_generator

kmeans_dir = os.path.join(os.path.dirname(__file__), 'data', 'kmeans')

num_clusters = 16 #TODO: save this value in model_dir
kmeans = KMeansClustering(num_clusters=num_clusters,
                          distance_metric=KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE,
                          mini_batch_steps_per_iteration = 100,
                          use_mini_batch=True,
                          model_dir=kmeans_dir)

def lab_to_labels(image_batch):
    cluster_centers = kmeans.cluster_centers()
    da = tf.expand_dims(image_batch[:,:,:,1],3) - cluster_centers[np.newaxis,np.newaxis,np.newaxis,:,0]
    db = tf.expand_dims(image_batch[:,:,:,2],3) - cluster_centers[np.newaxis,np.newaxis,np.newaxis,:,1]
    d = tf.square(da) + tf.square(db)
    return tf.argmin(d, 3)

def labels_to_lab(labels):
    cluster_centers = kmeans.cluster_centers()
    if labels.dtype in [np.float16, np.float32, np.float64]:
        l = tf.cast(tf.expand_dims(labels,-1), tf.float32)
        c = cluster_centers[np.newaxis,np.newaxis,np.newaxis,:,:]
        ab = tf.reduce_sum(l * c, 3)
    else:
        ab = tf.gather(cluster_centers, labels)
    l = tf.ones(tf.shape(ab)[:-1], tf.float32) * 75
    return tf.concat([tf.expand_dims(l,-1), ab], 3)

if __name__=='__main__':
    def preprocess(x):
        x = tf.image.resize_images(x, [32,32])
        return tf.reshape(x[:,:,:,1:], [-1,2])
    kmeans.train(lambda: create_batch_generator(1).map(preprocess))
    print kmeans.cluster_centers()

    import cv2
    sess = tf.Session()
    labels = np.arange(16).reshape([1,1,16])
    lab_img = sess.run(labels_to_lab(labels))[0]
    print lab_img
    bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    print bgr_img
    bgr_img = np.repeat(np.repeat(bgr_img, 100, 0), 30, 1)
    cv2.imshow('Color palette', bgr_img)
    cv2.waitKey(0)
