import pandas as pd 
from tqdm import tqdm 
import os 
path = './../data/train_downsampled.csv'
data = pd.read_csv(path)
# data.head()

cnt = {}
for i, row in tqdm(data.iterrows()):
  if row['image_id'] not in cnt:
      cnt[row['image_id']] = 0 
  if row['class_id'] != 14:
    cnt[row['image_id']] += 1

data2cls = pd.DataFrame() 
imgs = []
lbls = [] 
for img in tqdm(cnt.keys()):
  imgs.append(img)
  lbls.append(int(cnt[img] > 0)) 
data2cls['filename'] = imgs 
data2cls['label'] = lbls

os.makedirs('data', exist_ok=True) 
data2cls.to_csv('data/train_full.csv', index = False)