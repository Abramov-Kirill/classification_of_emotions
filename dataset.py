import torch
import torchaudio
import pandas as pd
import numpy as np

import time



class WavToMelDateset(torch.utils.data.Dataset):
    def __init__(self, config, path_wavs, path_file_csv, device, transform):
        self.sample_rate = config.sample_rate
        self.n_fft = config.n_fft
        self.win_length = config.win_length
        self.hop_length = config.hop_length
        self.f_min = config.f_min
        self.f_max = config.f_max
        self.n_mels = config.n_mels
        self.segment_size = config.segment_size

        self.path_wavs = path_wavs
        self.path_file_csv = path_file_csv

        self.df = pd.read_csv(self.path_file_csv)

        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        start_time = time.perf_counter()
        audio_path = self.path_wavs + '/' + self.df.iloc[item]['audio_path']
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        waveform = waveform.to(self.device)
        if waveform.size(1) >= self.segment_size:
            audio_start = waveform.size(1)//2 - self.segment_size//2
            waveform = waveform[:, audio_start:audio_start + self.segment_size]
        else:
            waveform = waveform.repeat((1, (48000 // waveform.shape[1]) + 1))[:,:48000]
            #waveform = torch.nn.functional.pad(waveform, (0, self.segment_size-waveform.shape[1]) , mode='constant')

        mel_specgram = self.transform(waveform)
        #print(time.perf_counter() - start_time)
        return mel_specgram, torch.tensor(self.df.iloc[item]['speaker_emo'], dtype=torch.long).to(self.device)






class MelDateset(torch.utils.data.Dataset):
    def __init__(self, path_spectrograms, path_file_csv, device):

        self.path_spectrograms = path_spectrograms
        self.path_file_csv = path_file_csv

        self.df = pd.read_csv(self.path_file_csv)

        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        start_time = time.perf_counter()
        mel_spectrogram_path = self.path_spectrograms + '/' + self.df.iloc[item]['audio_path']
        mel_spectrogram = torch.from_numpy(np.load(mel_spectrogram_path)).to(self.device)
        #print(time.perf_counter() - start_time)
        return mel_spectrogram, torch.tensor(self.df.iloc[item]['speaker_emo'], dtype=torch.long).to(self.device)






'''
    def getitem_full(self, item):
        audio_path = self.path_wavs + '/' + self.df.iloc[item]['audio_path']
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        waveform = waveform.to(self.device)
        if waveform.size(1) >= self.segment_size:
            initial_size = waveform.size(1)
            if initial_size % self.segment_size > 16000:
                waveform = nn.ConstantPad1d((0, 48000 - (initial_size % 48000)), 0)(waveform)
            else:
                waveform = waveform[:, :initial_size - (initial_size % self.segment_size)]
            waveform = torch.split(waveform, 48000, dim=1)
            waveform = torch.stack(waveform)

        else:
            waveform = waveform.repeat((1, (48000 // waveform.shape[1]) + 1))[:,:48000]
            waveform = waveform.unsqueeze(0)
            #waveform = torch.nn.functional.pad(waveform, (0, self.segment_size-waveform.shape[1]) , mode='constant')

        mel_specgram = self.transform(waveform)
        return mel_specgram, torch.tensor(self.df.iloc[item]['speaker_emo'], dtype=torch.long).to(self.device)

'''






















































