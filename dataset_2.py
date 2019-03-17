import wave
import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from config import Config

class FGPA_Dataset(Dataset):
    
    def __init__(self, path, filenames, labels, use_mfcc=False):
        super(FGPA_Dataset, self).__init__()
        self.dir = path
        # self.sr = 44100 if use_mfcc else 16000
        self.sr = Config.SAMPLING_RATE
        self.max_duration_in_sec = Config.MAX_AUDIO_LENGTH
        self.max_length = self.sr * self.max_duration_in_sec
        self.use_mfcc = use_mfcc
        self.n_mfccs = 40
        
        self.filenames = filenames
        self.labels = labels
        
        self.sequences = np.array([self.load_data_from(filename) for filename in filenames])
        print(self.sequences.shape)
        
    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]
    
    def __len__(self):
        return len(self.sequences)
    
    def load_data_from(self, filename):

        original_samples = self.read_waveform(filename)

        if len(original_samples) > self.max_length:
            max_offset = len(original_samples) - self.max_length
            offset = np.random.randint(max_offset)
            samples = original_samples[offset:(self.max_length+offset)]
        else:
            if self.max_length > len(original_samples):
                max_offset = self.max_length - len(original_samples)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            samples = np.pad(original_samples, (offset, self.max_length - len(original_samples) - offset), "constant")

        if self.use_mfcc:
            samples = librosa.feature.mfcc(samples, sr=self.sr, n_mfcc=self.n_mfccs)
        else:
            samples = self.normalization(samples)
            
        return samples
    
    def read_waveform(self, filename):
        return librosa.core.load(self.dir+filename, sr=self.sr,res_type='kaiser_fast')[0]

    def normalization(self, data):
        max_data = np.max(data)
        min_data = np.min(data)
        data = (data - min_data) / (max_data - min_data + 1e-6)
        return data - 0.5