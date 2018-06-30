import tensorflow as tf
import cv2
import numpy as np
from dataset import create_batch_generator
from clustering import lab_to_labels, labels_to_lab

sess = tf.Session()

gen = create_batch_generator(1).map(lab_to_labels).map(labels_to_lab).make_one_shot_iterator()
while True:
    lab_img = sess.run(gen.get_next())[0]
    bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    cv2.imshow('Image', bgr_img)
    cv2.waitKey(1)
