import json
import attrdict
from utils import *
from model import EmotionClassificationModel
import os
import shutil
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

path_for_checkpoints = 'check_points_2class'
if 'config.json' not in os.listdir(path_for_checkpoints):
    shutil.copy2('config.json', path_for_checkpoints + '/config.json')

with open(path_for_checkpoints + '/config.json') as f:
    data = f.read()
json_config = json.loads(data)
conf = attrdict.AttrDict(json_config)


transform = torchaudio.transforms.MelSpectrogram(sample_rate=conf.sample_rate, n_fft=conf.n_fft,
                                                         win_length=conf.win_length, hop_length=conf.hop_length,
                                                         f_min=conf.f_min, f_max=conf.f_max, n_mels=conf.n_mels)
transform.to(device)

if conf.type_dataset == 'mel_spectrograms':
    train_dataloader_4class, test_dataloader_4class, train_dataloader_2class, test_dataloader_2class = create_mel_data_loader(
        conf, device)
elif conf.type_dataset == 'wavs':
    train_dataloader_4class, test_dataloader_4class, train_dataloader_2clas, test_dataloader_2class = create_data_loader(
        conf, transform, device)

epochs = (0, 50)
count_class = conf.count_class

model_CNN = EmotionClassificationModel(conf=conf)
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_CNN.parameters(), lr=conf.learning_rate)
model_CNN.to(device)

print(model_CNN)
summary(model_CNN, (1, conf.n_mels, conf.segment_size // conf.hop_length), batch_size=1)

if count_class == 2:
    train_model(conf, model_CNN, criterion, optimizer, epochs, train_dataloader_2class, test_dataloader_2class,
                path_for_checkpoints, count_class)
elif count_class == 4:
    train_model(conf, model_CNN, criterion, optimizer, epochs, train_dataloader_4class, test_dataloader_4class,
                path_for_checkpoints, count_class)
