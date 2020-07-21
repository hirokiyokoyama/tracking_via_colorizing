# Tensorflow implementation of "Self-Supervised Tracking via Video Colorization"

Adapted to TensorFlow 2.x and Python 3. Old version (TensorFlow 1.x and Python 2) is available on the branch "tf1".
For details, see https://ai.googleblog.com/2018/06/self-supervised-tracking-via-video.html.

## Requirements
* youtube-dl
* ffmpeg
* cv2
* sklearn
* tensorflow >= 2.2.0

## Usage
### Training
1. python download_dataset.py (This downloads [kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) dataset and associated videos from youtube, which takes long time. I recommend to run it on background.)
1. python train.py

### Parameters in train.py
* NUM_REF: Number of reference frames.
* NUM_TARGET: Number of target frames.
* NUM_CLUSTERS: Number of color clusters.
* KMEANS_STEPS_PER_ITERATION: Steps per iteration for k-means algorithm.
* IMAGE_SIZE: Size of image as input of CNN.
* FEATURE_MAP_SIZE: Size of feature map. For CNN, number of rows and columns at the last layer. The network structure must be consistent with this value.
* MODEL_DIR: Directory where the model is to be saved.
* BATCH_SIZE: Size of mini-batches.

### Test
By following command, you can check if the trained model can perform tracking, which is not explicitly trained.

    python test_with_davis.py
    
This downloads [DAVIS2017](https://davischallenge.org/davis2017/code.html) dataset.
