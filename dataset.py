# Code adapted from https://www.kaggle.com/sunqpark/data-loader-for-pytorch-with-mfcc/code
import sys, os
import pandas as pd
from scipy.io import wavfile
import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch import Tensor

from config import Config

class FreesoundDataset(Dataset):
	def __init__(self, transform=None, mode="train"):
		# setting directories for data
		data_root = "../../../Storage/FSDKaggle2018_2"
		self.mode = mode

		if self.mode == "train":
			train_idx, validation_idx = next(iter(StratifiedKFold(n_splits=5).split(np.zeros_like(train_label_ids), train_label_ids)))
			self.train_idx = train_idx
			self.validation_idx = validation_idx
			self.data_dir = os.path.join(data_root, "audio_train")
			self.csv_file = pd.read_csv(os.path.join(data_root, "train.csv"))
		elif self.mode == "test":
			self.data_dir = os.path.join(data_root, "audio_test")
			self.csv_file = pd.read_csv(os.path.join(data_root, "sample_submission.csv"))

		# dict for mapping class names into indices. can be obtained by 
		# {cls_name:i for i, cls_name in enumerate(csv_file["label"].unique())}
		self.classes = {'Acoustic_guitar': 38, 'Applause': 37, 'Bark': 19, 'Bass_drum': 21, 'Burping_or_eructation': 28, 'Bus': 22, 'Cello': 4, 'Chime': 20, 'Clarinet': 7,'Computer_keyboard': 8, 'Cough': 17, 'Cowbell': 33, 'Double_bass': 29, 'Drawer_open_or_close': 36, 'Electric_piano': 34, 'Fart': 14, 'Finger_snapping': 40, 'Fireworks': 31, 'Flute': 16, 'Glockenspiel': 3, 'Gong': 26, 'Gunshot_or_gunfire': 6, 'Harmonica': 25, 'Hi-hat': 0, 'Keys_jangling': 9, 'Knock': 5, 'Laughter': 12, 'Meow': 35, 'Microwave_oven': 27, 'Oboe': 15, 'Saxophone': 1, 'Scissors': 24, 'Shatter': 30, 'Snare_drum': 10, 'Squeak': 23, 'Tambourine': 32, 'Tearing': 13, 'Telephone': 18, 'Trumpet': 2, 'Violin_or_fiddle': 39,  'Writing': 11}
		self.transform = transform
		self.max_length = Config.MAX_AUDIO_LENGTH * Config.SAMPLING_RATE
		
	def __len__(self):
		return self.csv_file.shape[0] 

	def __getitem__(self, idx):
		filename = self.csv_file["fname"][idx]
		
		rate, data = wavfile.read(os.path.join(self.data_dir, filename))
		
		if(rate != Config.SAMPLING_RATE):
			raise Exception(f"Sample rate: {rate} of file {filename} is not as expected.")

		# Random offset / Padding
		if data.size > self.max_length:
			max_offset = data.size - self.max_length
			offset = np.random.randint(max_offset)
			data = data[offset:(self.max_length+offset)]
		else:
			offset = 0
			if data.size < self.max_length:
				max_offset = self.max_length - data.size
				offset = np.random.randint(max_offset)
			data = np.pad(data, (offset, self.max_length - data.size - offset), "constant")

		if self.transform is not None:
			data = self.transform(data)

		if self.mode is "train":
			label = self.classes[self.csv_file["label"][idx]]
			return data, label
		elif self.mode is "test":
			return data


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	tsfm = transforms.Compose([
		lambda x: x.astype(np.float32) / np.max(x), # rescale to -1 to 1
		lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC (sampling rate = 44.1 kHz)
		lambda x: Tensor(x)
	])

	dataloader = DataLoader(
		FreesoundDataset(transform=tsfm, mode="train"), 
		batch_size=1,
		shuffle=True, 
		num_workers=0
	)

	for index, (data, label) in enumerate(dataloader):
		print(label.numpy())
		print(data.shape)
		plt.imshow(data.numpy()[0, :, :])
		plt.show()

		if index == 2:
			break