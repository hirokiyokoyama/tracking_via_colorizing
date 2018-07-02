import json
import os
import numpy as np
import cv2
import tensorflow as tf
from itertools import cycle

data_dir = os.path.join(os.path.dirname(__file__), 'data')
kinetics_dir = os.path.join(data_dir, 'kinetics_train')
kinetics_path = os.path.join(kinetics_dir, 'kinetics_train.json')
video_dir = os.path.join(data_dir, 'videos')
if __name__ != '__main__':
    if not os.path.exists(kinetics_path):
        raise ImportError('%s does not exist. Run dataset.py first.' % kinetics_path)
    kinetics = json.load(open(kinetics_path))

def create_ref_target_generator(num_ref=3, num_target=1, ref_skip=4, target_skip=4):
    skips = [ref_skip]*num_ref + [target_skip]*num_target
    skips[0] = 0
    
    def generate_frames():
        for key, entry in kinetics.iteritems():
            filename = os.path.join(video_dir, key+'.mp4')
            print 'Opening %s' % filename
            cap = cv2.VideoCapture(filename)
            frames = np.zeros((num_ref+num_target,
                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                               int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.float32)
            batches = 0
            for i, skip in cycle(enumerate(skips)):
                for _ in range(skip):
                    cap.read()
                ret, frame = cap.read()
                if not ret:
                    break
                frames[i] = cv2.cvtColor(np.float32(frame/255.), cv2.COLOR_BGR2LAB)
                if i == num_ref + num_target - 1:
                    if num_target > 0:
                        yield frames[:num_ref], frames[num_ref:]
                    else:
                        yield frames
                    batches += 1
            print 'Extracted %d batches from %s' % (batches, filename)
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
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)

    if not os.path.exists(kinetics_path):
        if not os.path.exists(kinetics_dir):
            os.mkdir(kinetics_dir)
        print 'Downloading kinetics dataset.'
        kinetics_url = 'https://deepmind.com/documents/193/kinetics_600_train%20(1).zip'
        kinetics_zip = os.path.join(data_dir, 'kinetics_train.zip')
        os.system('curl "%s" > %s' % (kinetics_url, kinetics_zip))
        os.system('unzip %s -d %s' % (kinetics_zip, kinetics_dir))
        os.remove(kinetics_zip)
    kinetics = json.load(open(kinetics_path))

    for key in kinetics.keys():
        entry = kinetics[key]
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
