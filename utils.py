import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import WavToMelDateset, MelDateset
import os
import torchaudio
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

ROOT_DIR = str(os.path.dirname(os.path.abspath(__file__)))

PATH_WAVS_TRAIN = os.path.join(ROOT_DIR, "waws/crowd_train")
PATH_WAVS_TEST =  os.path.join(ROOT_DIR, "waws/crowd_test")
PATH_CSV_TRAIN_4CLASS =  os.path.join(ROOT_DIR, "preparation_csv/train_balance_4class")
PATH_CSV_TEST_4CLASS = os.path.join(ROOT_DIR, "preparation_csv/test_balance_4class")
PATH_CSV_TRAIN_2CLASS = os.path.join(ROOT_DIR, "preparation_csv/train_balance_2class")
PATH_CSV_TEST_2CLASS = os.path.join(ROOT_DIR, "preparation_csv/test_balance_2class")

PATH_MEL_SPECTROGRAM_TRAIN = os.path.join(ROOT_DIR, "mel_spectrograms/train")
PATH_MEL_SPECTROGRAM_TEST =  os.path.join(ROOT_DIR, "mel_spectrograms/test")
PATH_MEL_SPECTROGRAM_CSV_TRAIN_4CLASS =  os.path.join(ROOT_DIR, "mel_spectrograms/train/train_balance_4class")
PATH_MEL_SPECTROGRAM_CSV_TEST_4CLASS = os.path.join(ROOT_DIR, "mel_spectrograms/test/test_balance_4class")
PATH_MEL_SPECTROGRAM_CSV_TRAIN_2CLASS = os.path.join(ROOT_DIR, "mel_spectrograms/train/train_balance_2class")
PATH_MEL_SPECTROGRAM_CSV_TEST_2CLASS = os.path.join(ROOT_DIR, "mel_spectrograms/test/test_balance_2class")



def train_model(conf, model, criterion, optimizer, epochs, train_dataloader, test_dataloader, path_for_checkpoints, count_class):
    for ep in tqdm(range(epochs[0], epochs[1])):
        total_train = 0
        correct_train = 0
        loss = None
        for mel_spec, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(mel_spec)
            labels_one_hot = nn.functional.one_hot(labels, count_class)
            labels_one_hot = torch.tensor(labels_one_hot, dtype=torch.float)

            loss = criterion(outputs, labels_one_hot)
            loss.backward()
            optimizer.step()


            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        print('\n')
        print('Epoch: ' + str(ep))
        print('Loss: ' + str(loss))
        print('Accuracy on the train set: %d %%' % (100 * correct_train / total_train))

        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for test_mel_spec, test_label in test_dataloader:
                outputs = model(test_mel_spec)
                _, predicted = torch.max(outputs.data, 1)
                total_test += test_label.size(0)
                correct_test += (predicted == test_label).sum().item()
            print('Accuracy on the test set: %d %%' % (100 * correct_test / total_test))

        torch.save(model.state_dict(), path_for_checkpoints + f'/ep_{ep}')
    pass


def create_data_loader(conf, transform, device):

    train_data_4class = WavToMelDateset(conf, PATH_WAVS_TRAIN, PATH_CSV_TRAIN_4CLASS, device, transform)
    train_dataloader_4class = DataLoader(train_data_4class, batch_size=conf.batch_size, shuffle=True)
    test_data_4class = WavToMelDateset(conf, PATH_WAVS_TEST, PATH_CSV_TEST_4CLASS, device, transform)
    test_dataloader_4class = DataLoader(test_data_4class, batch_size=conf.batch_size, shuffle=True)

    train_data_2class = WavToMelDateset(conf, PATH_WAVS_TRAIN, PATH_CSV_TRAIN_2CLASS, device, transform)
    train_dataloader_2class = DataLoader(train_data_2class, batch_size=conf.batch_size, shuffle=True)
    test_data_2class = WavToMelDateset(conf, PATH_WAVS_TEST, PATH_CSV_TEST_2CLASS, device, transform)
    test_dataloader_2class = DataLoader(test_data_2class, batch_size=conf.batch_size, shuffle=True)

    return train_dataloader_4class, test_dataloader_4class, train_dataloader_2class, test_dataloader_2class


def create_mel_data_loader(conf, device):
    train_data_4class = MelDateset(PATH_MEL_SPECTROGRAM_TRAIN, PATH_MEL_SPECTROGRAM_CSV_TRAIN_4CLASS, device)
    train_dataloader_4class = DataLoader(train_data_4class, batch_size=conf.batch_size, shuffle=True)
    test_data_4class = MelDateset(PATH_MEL_SPECTROGRAM_TEST, PATH_MEL_SPECTROGRAM_CSV_TEST_4CLASS, device)
    test_dataloader_4class = DataLoader(test_data_4class, batch_size=conf.batch_size, shuffle=True)

    train_data_2class = MelDateset(PATH_MEL_SPECTROGRAM_TRAIN, PATH_MEL_SPECTROGRAM_CSV_TRAIN_2CLASS, device)
    train_dataloader_2class = DataLoader(train_data_2class, batch_size=conf.batch_size, shuffle=True)
    test_data_2class = MelDateset(PATH_MEL_SPECTROGRAM_TEST, PATH_MEL_SPECTROGRAM_CSV_TEST_2CLASS, device)
    test_dataloader_2class = DataLoader(test_data_2class, batch_size=conf.batch_size, shuffle=True)

    return train_dataloader_4class, test_dataloader_4class, train_dataloader_2class, test_dataloader_2class


def test_model(conf, model, test_dataloader, check_points_path):
    check_points = os.listdir(check_points_path)
    for i in tqdm(check_points):
        if i == 'config.json':
            continue
        model.load_state_dict(torch.load(check_points_path + '/' + i))

        total_test = 0
        correct_test = 0
        with torch.no_grad():
            for test_mel_spec, test_label in test_dataloader:
                outputs = model(test_mel_spec)
                _, predicted = torch.max(outputs.data, 1)
                total_test += test_label.size(0)
                correct_test += (predicted == test_label).sum().item()
            print('Accuracy on the test set: %d %%' % (100 * correct_test / total_test))


def create_spectrograms(path_wavs, path_to_save, segment_size, transform):
    files_wav = os.listdir(path_wavs)
    for f_name in files_wav:
        waveform, sample_rate = torchaudio.load(path_wavs + '/' + f_name, normalize=True)

        if waveform.size(1) >= segment_size:
            audio_start = waveform.size(1)//2 - segment_size//2
            waveform = waveform[:, audio_start:audio_start + segment_size]
        else:
            #waveform = waveform.repeat((1, (segment_size // waveform.shape[1]) + 1))[:,segment_size]
            waveform = torch.nn.functional.pad(waveform, (0, segment_size-waveform.shape[1]) , mode='constant')

        mel_specgram = transform(waveform)
        np.save((path_to_save + '/' + f_name).replace('.wav', ''), mel_specgram)


def show_confusion_matrix(y_true, y_pred, names_class, x_label, y_label):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm,
                annot=True,
                fmt='g',
                xticklabels=names_class,
                yticklabels=names_class)
    plt.ylabel(y_label, fontsize=13)
    plt.xlabel(x_label, fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()

def accuracy_experts(df):
    labels = df['annotator_emo'].unique()
    labels = labels[labels != 'other']
    print(labels)
    acc = []
    for l in labels:
        count_eq = len(df.query(f'annotator_emo=="{l}" & speaker_emo=="{l}"'))
        count_noeq = len(df.query(f'annotator_emo!="{l}" & speaker_emo=="{l}"'))
        acc.append(count_eq / (count_eq + count_noeq))
    print(acc)
    print(sum(acc) / len(acc))