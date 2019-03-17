import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from .flatten import Flatten

class Model_1(nn.Module):
	def __init__(self):
		super().__init__()
		self.name = "model_1"

		self.block_1 = nn.Sequential(
			nn.Conv1d(1, 16, kernel_size=3),
			nn.MaxPool1d(3),
			nn.BatchNorm1d(16),
			nn.ReLU()
		)

		self.block_2 = nn.Sequential(
			nn.Conv1d(16, 32, kernel_size=3),
			nn.MaxPool1d(3),
			nn.BatchNorm1d(32),
			nn.ReLU()
		)
		self.block_3 = nn.Sequential(
			nn.Conv1d(32, 64, kernel_size=3),
			nn.MaxPool1d(3),
			nn.BatchNorm1d(64),
			nn.ReLU()		
		)
		self.block_4 = nn.Sequential(
			nn.Conv1d(64, 128, kernel_size=3),
			nn.MaxPool1d(3),
			nn.BatchNorm1d(128),
			nn.ReLU()
		)

		self.block_5 = nn.Sequential(
			nn.Conv1d(128, 256, kernel_size=3),
			nn.MaxPool1d(3),
			nn.BatchNorm1d(256),
			nn.ReLU()	
		)

		self.block_6 = nn.Sequential(
			nn.Conv1d(256, 528, kernel_size=3),
			nn.MaxPool1d(3),
			nn.BatchNorm1d(528),
			nn.ReLU()	
		)

		self.block_7 = nn.Sequential(
			nn.Conv1d(528, 1024, kernel_size=3),
			nn.MaxPool1d(3),
			nn.BatchNorm1d(1024),
			nn.ReLU()	
		)

		self.block_8 = nn.Sequential(
			nn.Conv1d(1024, 1024, kernel_size=3),
			nn.MaxPool1d(2),
			nn.ReLU()
		)

		self.flatten = Flatten()

		self.dense_layers = nn.Sequential(
			nn.Linear(5120, 526),
			nn.Dropout(0.1),
			nn.ReLU(),
			nn.Linear(526, 254),
			nn.Dropout(0.1),
			nn.ReLU(),
			nn.Linear(254, 41)
		)

		# Grouping togheter layers
		self.conv_layers = nn.Sequential(
			self.block_1,
			self.block_2,
			self.block_3,
			self.block_4,
			self.block_5,
			self.block_6,
			self.block_7,
			self.block_8
		)

	def forward(self, x):
		out = self.conv_layers(x)
		out = self.flatten(out)
		# print("new Shape:", out.shape)
		# raise Exception("You shall not pass.")
		out = self.dense_layers(out)
		return out