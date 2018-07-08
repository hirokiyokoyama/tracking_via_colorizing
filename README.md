# Tensorflow implementation of "Self-Supervised Tracking via Video Colorization"

For details, see https://ai.googleblog.com/2018/06/self-supervised-tracking-via-video.html.

## Requirements
* *Latest version* of youtube-dl (see https://rg3.github.io/youtube-dl/download.html)
* ffmpeg
* cv2
* sklearn
* and, of course, tensorflow

## Usage
### Training
1. python dataset.py (this downloads [kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) dataset and associated videos from youtube, which takes long time)
1. ~~python clustering.py (clusters colors that appear in the videos using kmeans)~~(currently color clusters are updated while training)
1. python train.py

Step 1 lasts almost forever! You can go to step 2 and 3 while running dataset.py. Missing videos will be ignored.
To watch the progress of training, run the following command and browse http://\<its IP address\>:6006.

    tensorboard --logdir=data/model

### Test
By following command, you can check if the trained model can perform tracking, which is not explicitly trained.

    python test_with_davis.py <ckpt file> [image set name]
    
This downloads [DAVIS2017](https://davischallenge.org/davis2017/code.html) dataset.
Note that the ckpt file path must ends with "-<# steps>" if any. E.g. "/path/to/ckpt/model.ckpt-1000" for one at 1000th step.
Ignore the suffixes like ".data-00000-of-00001".
You can specify the name of video ("aerobatics", "car-raceas", etc.) as the second argument, or it will be randomly selected.
