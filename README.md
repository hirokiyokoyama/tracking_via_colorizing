# Tensorflow implementation of "Self-Supervised Tracking via Video Colorization"

For details, see https://ai.googleblog.com/2018/06/self-supervised-tracking-via-video.html.

## Requirements
* *Latest version* of youtube-dl (see https://rg3.github.io/youtube-dl/download.html)
* ffmpeg

## Usage
1. python dataset.py (this downloads kinetics dataset and associated videos from youtube, which takes long time)
1. python clustering.py (clusters colors that appear in the videos using kmeans)
1. python train.py
