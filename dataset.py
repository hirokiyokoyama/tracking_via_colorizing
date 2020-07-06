import tensorflow as tf
import numpy as np
import cv2
import os
import json
import datetime
import subprocess

_kinetics_url = 'https://storage.googleapis.com/deepmind-media/Datasets/kinetics600.tar.gz'
_davis_url = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip'

class VideoDownloader:
  def __init__(self, preferred_size=None):
    import youtube_dl
    self._ydl = youtube_dl.YoutubeDL({'format': 'mp4'})
    self._preferred_size = preferred_size

  def download(self, url, segment, filename):
    try:
      formats = self._ydl.extract_info(url, download=False)['formats']
    except:
      return 'extract_info_failed'

    formats = list(filter(lambda x: x['width'] and x['height'], formats))
    if not formats:
      return 'no_format_available'

    key = lambda x: min(x['width'], x['height'])
    if self._preferred_size:
      formats_larger = list(filter(lambda x: key(x) >= self._preferred_size, formats))
      formats_larger.sort(key=key, reverse=False)
      formats_smaller = list(filter(lambda x: key(x) < self._preferred_size, formats))
      formats_smaller.sort(key=key, reverse=True)
      formats = formats_larger + formats_smaller
    else:
      formats.sort(key=key, reverse=True)

    for format in formats:
      begin, end = map(lambda x: datetime.timedelta(seconds=x), segment)
      p = subprocess.Popen([
                            'ffmpeg', '-i', format["url"],
                            '-ss', str(begin), '-to', str(end),
                            '-c', 'copy', filename],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
      try:
        p.wait(timeout=60)
      except subprocess.TimeoutExpired:
        p.kill()
        return 'download_failed'
      if p.returncode == 0:
        return 'success'
    return 'download_failed'

def download_kinetics(url, dest_dir):
  filename = url.split('/')[-1]
  if not filename.endswith('.tar.gz'):
    raise ValueError(f'Unsupported URL: {url}')
  filepath = os.path.join(dest_dir, filename)
  if not os.path.exists(dest_dir):
     os.mkdir(dest_dir)

  if os.system(f'curl -L -o {filepath} {url}') != 0:
    raise Exception('Download failed.')
  if os.system(f'tar -zxvf {filepath} -C {dest_dir}') != 0:
    raise Exception('Extraction failed.')
  os.remove(filepath)

def load_kinetics(base_dir, split, download=True):
  json_file = os.path.join(base_dir, 'kinetics600', split+'.json')
  if not os.path.exists(json_file):
    if not download:
      return None
    download_kinetics(_kinetics_url, base_dir)
    
  with open(json_file) as f:
    return json.load(f)

def download_kinetics_videos(kinetics, dest_dir, preferred_size=256):
  dl = VideoDownloader(preferred_size=preferred_size)

  for key, entry in kinetics.items():
    filename = os.path.join(dest_dir, key+'.mp4')
    if os.path.exists(filename):
      continue
    result = dl.download(entry['url'], entry['annotations']['segment'], filename)
    print(result)

def download_davis(url, dest_dir):
  filename = url.split('/')[-1]
  if not filename.endswith('.zip'):
    raise ValueError(f'Unsupported URL: {url}')
  filepath = os.path.join(dest_dir, filename)
  if not os.path.exists(dest_dir):
     os.mkdir(dest_dir)
  
  if os.system(f'curl -L -o {filepath} {url}') != 0:
    raise Exception('Download failed.')
  if os.system(f'unzip {filepath} -d {dest_dir}') != 0:
    raise Exception('Extraction failed.')
  os.remove(filepath)
