import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from .flatten import Flatten

class Baseline(nn.Module):
	def __init__(self):
		super().__init__()
		self.name = "model_1"

		self.conv1 = nn.Conv1d(1, 16, kernel_size=9)
		self.max_pool1 = nn.MaxPool1d(3)
		self.dropout1 = nn.Dropout(0.1)
		self.relu1 = nn.ReLU()

		self.conv2 = nn.Conv1d(16, 16, kernel_size=3)
		self.max_pool2 = nn.MaxPool1d(7)
		self.dropout2 = nn.Dropout(0.1)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv1d(16, 32, kernel_size=3)
		self.max_pool3 = nn.MaxPool1d(9)
		self.dropout3 = nn.Dropout(0.1)
		self.relu3 = nn.ReLU()

		self.flatten = Flatten()
		self.dense_layers = nn.Sequential(
			nn.Linear(29856, 256),
			nn.Dropout(0.5),
			nn.Linear(256, 41)
		)

		# Grouping togheter layers
		self.conv_layers = nn.Sequential(
			self.conv1,
			self.max_pool1,
			self.dropout1,
			self.relu1,
			self.conv2,
			self.max_pool2,
			self.dropout2,
			self.relu2,
			self.conv3,
			self.max_pool3,
			self.dropout3,
			self.relu3
		)

	def forward(self, x):
		out = self.conv_layers(x)
		out = self.flatten(out)
		out = self.dense_layers(out)
		return out