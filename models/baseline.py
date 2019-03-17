import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from .flatten import Flatten

class Baseline(nn.Module):
	def __init__(self, inplace=True, training=True):
		super().__init__()
		self.name = "model_1"
		self.inplace = inplace
		self.training = training

		self.conv11 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9)
		self.conv12 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=9)

		self.conv21 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
		self.conv22 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)

		self.conv31 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
		self.conv32 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)

		self.conv41 = nn.Conv1d(in_channels=32, out_channels=256, kernel_size=3)
		self.conv42 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)

		self.l1 = nn.Linear(256, 64)
		self.l2 = nn.Linear(64, 1028)
		self.l3 = nn.Linear(1028, 41)

		self.flatten = Flatten()


	def forward(self, x):
		x = F.relu(self.conv11(x), inplace=self.inplace)
		x = F.relu(self.conv12(x), inplace=self.inplace)
		x = F.max_pool1d(x, kernel_size=16)
		x = F.dropout(x, 0.1, self.training, self.inplace)

		x = F.relu(self.conv21(x), inplace=self.inplace)
		x = F.relu(self.conv22(x), inplace=self.inplace)
		x = F.max_pool1d(x, kernel_size=4)
		x = F.dropout(x, 0.1, self.training, self.inplace)

		x = F.relu(self.conv31(x), inplace=self.inplace)
		x = F.relu(self.conv32(x), inplace=self.inplace)
		x = F.max_pool1d(x, kernel_size=4)
		x = F.dropout(x, 0.1, self.training, self.inplace)

		x = F.relu(self.conv41(x), inplace=self.inplace)
		x = F.relu(self.conv42(x), inplace=self.inplace)
		x = F.max_pool1d(x, kernel_size=119)
		x = F.dropout(x, 0.2, self.training, self.inplace)

		x = self.flatten(x)
		# print("new Shape:", x.shape)
		# raise Exception("You shall not pass.")
		x = F.relu(self.l1(x), inplace=self.inplace)
		x = F.relu(self.l2(x), inplace=self.inplace)
		x = F.softmax(x, dim=1)
		return x