import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from .flatten import Flatten

class Model_3(nn.Module):
	def __init__(self):
		super().__init__()
		self.name = "model_3"

		self.conv_block_1 = nn.Sequential(
			nn.Conv1d(1, 16, kernel_size=3),
			nn.MaxPool1d(3),
			nn.BatchNorm1d(16),
			nn.ReLU()
		)

		self.rnn_block_2 = nn.Sequential(
			nn.LSTM(
				input_size=, 
				hidden_size=, 
				num_layers=, 
				batch_first=, 
				bidirectional=True
			)
		)

		self.linear_block_3 = nn.Sequential(
            nn.Linear(832, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 41)
        )

	def forward(input):
		out = self.conv_block_1(input)
		out = self.rnn_block_2(out)
		out = self.linear_block_3(out)
		return out