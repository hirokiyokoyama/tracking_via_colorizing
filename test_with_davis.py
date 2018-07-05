import tensorflow as tf
import cv2
import numpy as np
import os
from itertools import product
from nets import colorizer

data_dir = os.path.join(os.path.dirname(__file__), 'data')
davis_dir = os.path.join(data_dir, 'DAVIS')
annotations_dir = os.path.join(davis_dir, 'Annotations', '480p')
images_dir = os.path.join(davis_dir, 'JPEGImages', '480p')

if not os.path.exists(davis_dir):
    print 'DAVIS dataset not found. Downloading.'
    url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip'
    zip_file = os.path.join(data_dir, 'DAVIS-2017-test-dev-480p.zip')
    os.system('curl "%s" > %s' % (url, zip_file))
    os.system('unzip %s -d %s' % (zip_file, data_dir))
    os.remove(zip_file)

_image_sets_file = os.path.join(davis_dir, 'ImageSets', '2017', 'test-dev.txt')
image_sets = map(lambda x: x.strip(), open(_image_sets_file).readlines())

def get_annotation(image_set):
    img = cv2.imread(os.path.join(annotations_dir, image_set, '00000.png'))
    mask = np.zeros(img.shape[:2], np.int32)
    color_dict = {0:0}
    colors = [np.array([0,0,0], np.uint8)]
    count = 1
    for i, j in product(xrange(img.shape[0]), xrange(img.shape[1])):
        pixel = img[i,j]
        _pixel = pixel[0] | pixel[1] << 8 | pixel[2] << 16
        if _pixel in color_dict:
            mask[i,j] = color_dict[_pixel]
        else:
            mask[i,j] = count
            color_dict[_pixel] = count
            colors.append(pixel)
            count += 1
    return mask, np.array(colors)

def one_hot(x, n):
    y = np.zeros(x.shape+(n,), dtype=np.float32)
    for i in xrange(n):
        y[np.where(x==i)+(i,)] = 1
    return y

def get_images(image_set):
    img_dir = os.path.join(images_dir, image_set)
    img_files = os.listdir(img_dir)
    img_files.sort()
    return np.array([cv2.imread(os.path.join(img_dir, f)) for f in img_files])

if __name__=='__main__':
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        print 'Usage: %s model_path [davis_image_set]' % sys.argv[0]
        quit()

    # load images
    if len(sys.argv) > 2 and sys.argv[2] in image_sets:
        image_set = sys.argv[2]
    else:
        image_set = np.random.choice(image_sets)
        print 'Using "%s" in DAVIS2017 dataset. To use another one, specify the name as the command line argument.' % image_set
    annotation_large, colors = get_annotation(image_set)
    colors = colors / 255.
    n_colors = colors.shape[0]
    annotation_large = one_hot(annotation_large, n_colors)
    annotation = cv2.resize(annotation_large, (32,32))
    
    all_images = get_images(image_set) / 255.
    all_images_gray = all_images.mean(3)
    all_images_gray = np.array([cv2.resize(img, (256,256)) for img in all_images_gray])
    all_images_gray = np.repeat(np.expand_dims(all_images_gray,-1), 3, -1)

    # load model
    with tf.Graph().as_default() as graph:
        saver = tf.train.import_meta_graph(model_path+'.meta')

        images = graph.get_tensor_by_name('images:0')
        features = graph.get_tensor_by_name('features:0')
        annotations = tf.placeholder(tf.float32, shape=[3,32,32,n_colors])
        
        end_points = colorizer(features[0:3], annotations, features[3:4])
        predictions = end_points['predictions']
        temperature = end_points['temperature']

    def apply_mask(image, mask):
        return mask[:,:,0:1] * image + (np.expand_dims(mask[:,:,1:], -1) * colors[1:].reshape(1,1,-1,3)).sum(2)
    
    cv2.imshow('Reference frame', apply_mask(all_images[0], annotation_large))
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, model_path)

        _annotations = np.expand_dims(annotation, 0).repeat(3, 0)
        for t in xrange(1, all_images.shape[0]):
            _images = all_images_gray[[max(t-3,0),max(t-2,0),t-1,t]]
            _predictions = sess.run(predictions, {images: _images,
                                                  annotations: _annotations,
                                                  temperature: 0.5})
            _annotations = np.roll(_annotations, -1, 0)
            _annotations[-1] = _predictions[0]
            mask = cv2.resize(_predictions[0], (all_images.shape[2], all_images.shape[1]))
            cv2.imshow('Target frame', apply_mask(all_images[t], mask))
            cv2.waitKey(1)
