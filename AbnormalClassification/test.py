import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as tvtf
from tqdm import tqdm
import yaml 
from datasets.image_folder import ImageFolderDataset
from utils.getter import get_instance
from utils.device import move_to
import pprint
import argparse
import pandas as pd 

# parser = argparse.ArgumentParser()
# parser.add_argument('-d', type=str,
#                     help='path to the folder of query images')
# parser.add_argument('-w', type=str,
#                     help='path to weight files')
# parser.add_argument('-g', type=int, default=None,
#                     help='(single) GPU to use (default: None)')
# parser.add_argument('-b', type=int, default=64,
#                     help='batch size (default: 64)')
# parser.add_argument('-o', type=str, default='test.csv',
#                     help='output file (default: test.csv)')
# args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--gpus', default=None)
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

config_path = args.config
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

assert config is not None, "Do not have config file!"

pprint.PrettyPrinter(indent=2).pprint(config)

config['gpus'] = args.gpus
config['debug'] = args.debug


# Device
dev_id = 'cuda:{}'.format(args.gpus) \
    if torch.cuda.is_available() and args.gpus is not None \
    else 'cpu'
device = torch.device(dev_id)

pretrained_path = config["pretrained"]

pretrained = None
if (pretrained_path != None):
    pretrained = torch.load(pretrained_path, map_location=dev_id)
    for item in ["model"]:
        config[item] = pretrained["config"][item]

# 2: Define network
model = get_instance(config['model']).to(device)

# Train from pretrained if it is not None
if pretrained is not None:
    model.load_state_dict(pretrained['model_state_dict'])
    

# Load data
tfs = tvtf.Compose([
    tvtf.Resize((224, 224)),
    tvtf.ToTensor(),
    tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
])


dataset = ImageFolderDataset(config['dataset']['csv'])
dataloader = DataLoader(dataset, batch_size=config['dataset']['batch_size'], num_workers = 8)

label = []
conff = []
filename = []
df = dataset.df
for a, b in df.iterrows():
    filename.append(b['filename'][b['filename'].rfind('/') + 1:])

probs_ls = [] 
  
with torch.no_grad():
    #out = [('filename', 'prediction', 'confidence')]
    model.eval()
    for i, (imgs, fns) in enumerate(tqdm(dataloader)):
        imgs = move_to(imgs, device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)
        for lbl, conf, prob in zip(preds, confs, probs):
            label.append(int(lbl))
            conff.append(conf.cpu().numpy())
            probs_ls.append(prob.cpu().numpy())
        #break
with open(config['id'] + '.csv', "w") as f:
  print("filename,label,conf,0,1,2,3,4,5,6,7,8",file = f)
  for i in range(len(label)):
    print(filename[i], label[i], conff[i], end = ',', sep = ',', file = f)
    for x in probs_ls[i]:
      print('{:05f}'.format(x), file = f, end = ',')
    print(file = f)
  f.close()  

with open(config['id'] + '.txt', "w") as f:
  for i in range(len(label)):
    lbl = label[i] if conff[i] >= config['threshold'] else 0
    print(filename[i], lbl, sep = '\t', file = f)
  f.close()  
