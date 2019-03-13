import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from .flatten import Flatten

class Baseline(nn.Module):
	def __init__(self):
		super().__init__()
		self.name = "baseline"

		self.conv = nn.Conv1d(1, 5, kernel_size=5)
		self.flatten = Flatten()
		self.linear = nn.Linear(
			5 * (Config.MAX_AUDIO_LENGTH * Config.SAMPLING_RATE - 4),
			41
		)

	def forward(self, x):
		# print("\n")
		# print(x.shape)
		out = self.conv(x)
		out = F.relu(out)
		# print(out.shape)
		out = self.flatten(out)
		# print(out.shape)
		out = self.linear(out)
		# out = F.softmax(out)
		return out