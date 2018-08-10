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

### Parameters in train.py
* NUM_REF: Number of reference frames.
* NUM_TARGET: Number of target frames.
* NUM_CLUSTERS: Number of color clusters.
* KMEANS_STEPS_PER_ITERATION: Steps per iteration for k-means algorithm.
* FEATURE_DIM: Dimension of feature space. For CNN, number of channels at the last layer.
* LEARNING_RATE: Function of training step that returns learning rate.
* WEIGHT_DECAY: Decay value of regularizer for filters in convolutional layers.
* BATCH_NORM_DECAY: Decay rate of moving average for batch normalization.
* BATCH_RENORM_DECAY: Decay rate of moving average for batch renormalization.
* IMAGE_SIZE: Size of image as input of CNN.
* FEATURE_MAP_SIZE: Size of feature map. For CNN, number of rows and columns at the last layer. The network structure must be consistent with this value.
* USE_CONV3D: Whether to perform spatio-temporal convolutions or only spatial convolutions.
* BATCH_RENORM_RMAX: Function of training step that returns rmax for batch renormalization.
* BATCH_RENORM_DMAX: Function of training step that returns dmax for batch renormalization.
* LOSS_WEIGHTING_SHARPNESS: Sharpness of weighting loss values to avoid samples with color inconsistency. Should be in (0,0.5].
* MODEL_DIR: Directory where the model is to be saved.
* USE_HISTORY: Whether or not to use prioritized replay buffer.
* INITIAL_WEIGHT: Initial priority of samples to be added into the replay buffer.
* BATCH_SIZE: Number of samples to be chosen from the replay buffer.
* TRAIN_INTERVAL: 
* HISTORY_CAPACITY: Capacity of the replay buffer.
* MIN_HISTORY_SIZE: Size of the replay buffer at which the training step begins.
* HISTORY_DEVICE: Device where variables and operations involving the replay buffer to be placed.

### Test
By following command, you can check if the trained model can perform tracking, which is not explicitly trained.

    python test_with_davis.py <ckpt file> [image set name]
    
This downloads [DAVIS2017](https://davischallenge.org/davis2017/code.html) dataset.
Note that the ckpt file path must ends with "-<# steps>" if any. E.g. "/path/to/ckpt/model.ckpt-1000" for one at 1000th step.
Ignore the suffixes like ".data-00000-of-00001".
You can specify the name of video ("aerobatics", "car-raceas", etc.) as the second argument, or it will be randomly selected.
