#!/usr/bin/env python3

import dataset
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
VIDEO_SIZE = 256

if __name__=='__main__':
  if len(sys.argv) > 1:
    split = sys.argv[1]
  else:
    split = 'train'
    
  kinetics = dataset.load_kinetics(DATA_DIR, split, donwload=True)
  video_dir = os.path.join(DATA_DIR, 'videos_'+str(VIDEO_SIZE))
  download_kinetics_videos(
      kinetics, video_dir, preferred_size=VIDEO_SIZE)
