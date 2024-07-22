import json
import attrdict
from utils import *
from model import EmotionClassificationResNet
import shutil
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

path_for_checkpoints = 'check_points_2class_3_6'
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

epochs = (0, 100)
count_class = conf.count_class

model = EmotionClassificationResNet(num_classes=count_class, conf=conf)
#model.load_state_dict(torch.load('check_points_4_class_block1_ch32/ep_16'))
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
model.to(device)

print(model)
summary(model, (1, conf.n_mels, conf.segment_size // conf.hop_length), batch_size=1)

if count_class == 2:
    train_model(conf, model, criterion, optimizer, epochs, train_dataloader_2class, test_dataloader_2class,
                path_for_checkpoints, count_class)
elif count_class == 4:
    train_model(conf, model, criterion, optimizer, epochs, train_dataloader_4class, test_dataloader_4class,
                path_for_checkpoints, count_class)
