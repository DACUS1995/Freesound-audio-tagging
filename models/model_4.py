import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from .flatten import Flatten

LSTM_LAYERS = 2
NUM_DIRECTIONS = 2 # bidirectional
HIDDEN_SIZE = 832

class Model_4(nn.Module):
	def __init__(self):
		super().__init__()
		self.name = "model_4"

		self.h_n = None
		self.c_n = None

		self.conv_block_1 = nn.Sequential(
			nn.Conv2d(1, 6, kernel_size=3, padding=2),
			nn.BatchNorm2d(6),
			nn.MaxPool2d(2, padding=1),
			nn.ReLU(),
			nn.Conv2d(6, 12, kernel_size=3, padding=2),
			nn.BatchNorm2d(12),
			nn.MaxPool2d(2, padding=1),
			nn.ReLU(),
			nn.Conv2d(12, 6, kernel_size=3, padding=2),
			nn.BatchNorm2d(6),
			nn.MaxPool2d(2, padding=1),
			nn.ReLU()
		)

		self.rnn_block_2 = nn.Sequential(
			nn.LSTM(
				input_size=6 * 8,
				hidden_size=HIDDEN_SIZE,
				num_layers=LSTM_LAYERS,
				batch_first=True,
				bidirectional=True,
				dropout=0.05
			)
		)

		self.linear_block_3 = nn.Sequential(
			nn.Linear(HIDDEN_SIZE * NUM_DIRECTIONS, 256),
			nn.Dropout(0.5),
			nn.Linear(256, 41)
		)

		self.flatten = Flatten()

	def forward(self, input):
		out = self.conv_block_1(input.view(input.shape[0], 1, input.shape[1], input.shape[2]))

		h_current = None
		c_current = None

		out = out.view(out.shape[0], out.shape[3], -1)

		# if self.h_n == None and self.c_n == None:
		# 	out, (h_current, c_current) = self.rnn_block_2(out)
		# else:
		out, (h_current, c_current) = self.rnn_block_2(out)
		self.h_n = h_current
		self.c_n = c_current

		out = h_current.view(Config.BATCH_SIZE, LSTM_LAYERS, NUM_DIRECTIONS, HIDDEN_SIZE)
		out = out[:, LSTM_LAYERS - 1, :, :]
		out = out.view(out.size(0), -1)

		out = self.linear_block_3(out)
		return out