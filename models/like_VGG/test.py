from utils import *
import json
import attrdict
from utils import *
from model import EmotionClassificationModel
import os
import shutil
from torchsummary import summary
from model import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

path_for_checkpoints = 'check_points_4class_ch32'
with open(path_for_checkpoints + '/config.json') as f:
    data = f.read()
json_config = json.loads(data)
conf = attrdict.AttrDict(json_config)

model = EmotionClassificationModel(conf)


train_dataloader_4class, test_dataloader_4class, train_dataloader_2class, test_dataloader_2class = \
    (create_mel_data_loader(conf, device))


test_model(conf, model, test_dataloader_4class, path_for_checkpoints)