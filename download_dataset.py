#!/usr/bin/env python3

import dataset
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
VIDEO_SIZE = 256

def main(split = 'train'):
  split = split or 'train'
    
  if not os.path.exists(DATA_DIR):
     os.mkdir(DATA_DIR)
  kinetics = dataset.load_kinetics(DATA_DIR, split, download=True)
  video_dir = os.path.join(DATA_DIR, 'videos_'+str(VIDEO_SIZE))
  dataset.download_kinetics_videos(
      kinetics, video_dir, preferred_size=VIDEO_SIZE)

if __name__=='__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1])
  else:
    main()
