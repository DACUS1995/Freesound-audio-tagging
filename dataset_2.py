import wave
import numpy as np
import librosa
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch import Tensor
from torchvision import transforms

from config import Config

class FGPA_Dataset(Dataset):
	
	def __init__(self, path, filenames, labels, use_mfcc=False):
		super(FGPA_Dataset, self).__init__()
		self.dir = path
		self.sr = 44100 if use_mfcc else 16000
		# self.sr = Config.SAMPLING_RATE
		self.max_duration_in_sec = Config.MAX_AUDIO_LENGTH
		self.max_length = self.sr * self.max_duration_in_sec
		self.use_mfcc = use_mfcc
		self.n_mfccs = 40
		
		self.filenames = filenames
		self.labels = labels

		self.transform = None

		if self.use_mfcc == True:
			self.transform = transforms.Compose([
				lambda x: x.astype(np.float32) / (np.max(x) if np.max(x) > sys.float_info.epsilon else sys.float_info.epsilon), # rescale to -1 to 1
				lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC (sampling rate = 44.1 kHz)
				lambda x: Tensor(x)
			])
		else:
			pass
			# self.sequences = np.array([self.load_data_from(filename) for filename in filenames])

		
		
	def __getitem__(self, index):
		if self.use_mfcc == True:
			return self.load_data_from(self.filenames[index]), self.labels[index]
		else:
			return self.load_data_from(self.filenames[index]), self.labels[index]
			# return self.sequences[index], self.labels[index]
	
	def __len__(self):
		return len(self.filenames)
	
	def load_data_from(self, filename):

		original_samples = self.read_waveform(filename)
		original_samples = self.remove_silence(original_samples)

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
		
		if self.use_mfcc and self.transform is not None:
			samples = self.transform(samples)
		else:
			samples = self.normalization(samples)
		
		return samples
	
	def read_waveform(self, filename):
		return librosa.core.load(self.dir+filename, sr=self.sr, res_type='kaiser_fast')[0]

	def normalization(self, data):
		max_data = np.max(data)
		min_data = np.min(data)
		data = (data - min_data) / (max_data - min_data + 1e-6)
		return data - 0.5

	def remove_silence(self, audio_segment):
		return librosa.effects.trim(audio_segment)
