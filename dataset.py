import json
import os
import numpy as np
import cv2
import tensorflow as tf

kinetic_path = os.path.join(os.path.dirname(__file__), 'data', 'kinetics_train', 'kinetics_train.json')
video_dir = os.path.join(os.path.dirname(__file__), 'data', 'videos')
kinetic = json.load(open(kinetic_path))

def create_ref_target_generator(num_ref=3, num_target=5):
    def generate_frames():
        for key, entry in kinetic.iteritems():
            filename = os.path.join(video_dir, key+'.mp4')
            print 'Opening %s' % filename
            cap = cv2.VideoCapture(filename)
            frames = None
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frames is None:
                    frames = np.zeros((num_ref+num_target,)+frame.shape, dtype=np.float32)
                frames[i] = cv2.cvtColor(np.float32(frame/255.), cv2.COLOR_BGR2LAB)
                i += 1
                if i == num_ref+num_target:
                    i = 0
                    if num_target > 0:
                        yield frames[:num_ref], frames[num_ref:]
                    else:
                        yield frames
            print 'Closing %s' % filename
    if num_target > 0:
        types = tf.float32, tf.float32
        shapes = tf.TensorShape([num_ref,None,None,3]), tf.TensorShape([num_target,None,None,3])
    else:
        types = tf.float32
        shapes = tf.TensorShape([num_ref,None,None,3])
    return tf.data.Dataset.from_generator(generate_frames, types, shapes)

def create_batch_generator(batch_size):
    return create_ref_target_generator(num_ref=batch_size, num_target=0)

if __name__=='__main__':
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)

    for key in kinetic.keys():
        entry = kinetic[key]
        url = entry['url']
        path1 = os.path.join(video_dir, '_'+key+'.mp4')
        path2 = os.path.join(video_dir, key+'.mp4')
        if os.path.exists(path2):
            print 'Skipping existing video "%s".' % key
        else:
          try:
            print 'Downloading video "%s".' % key
            command = ['youtube-dl',
                       '--quiet', '--no-warnings',
                       '-f', 'mp4',
                       '-o', '"%s"' % path1,
                       '"%s"' % url]
            os.system(' '.join(command))
            command = ['ffmpeg',
                       '-i', '"%s"' % path1,
                       '-t', '%f' % entry['duration'],
                       '-ss', '%f' % entry['annotations']['segment'][0],
                       '-strict', '-2',
                       '"%s"' % path2]
            os.system(' '.join(command))
            os.remove(path1)
          except:
            print 'Error with video "%s". Skipping.' % key
