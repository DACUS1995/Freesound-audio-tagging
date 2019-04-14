import torch
import torch.nn as nn
import torch.nn.functional as F

from models.flatten import Flatten

class Model_2(nn.Module):

	def __init__(self):
		super().__init__()
		self.name = "model_2"

		self.conv_layer1 = nn.Conv2d(
			in_channels = 1,
			out_channels = 32,
			kernel_size = (4,10)
		)
		self.batch_norm1 = nn.BatchNorm2d(32)
		self.max_pool1 = nn.MaxPool2d(2)
		self.relu1 = nn.ReLU()
		
		self.conv_layer2 = nn.Conv2d(
			in_channels = 32,
			out_channels = 32,
			kernel_size = (4,10)
		)
		self.batch_norm2 = nn.BatchNorm2d(32)
		self.max_pool2 = nn.MaxPool2d(2)
		self.relu2 = nn.ReLU()
		
		self.conv_layer3 = nn.Conv2d(
			in_channels = 32,
			out_channels = 32,
			kernel_size = (4,10)
		)
		self.batch_norm3 = nn.BatchNorm2d(32)
		self.max_pool3 = nn.MaxPool2d(2)
		self.relu3 = nn.ReLU()

		self.flatten = Flatten()
		
		self.conv_layers = nn.Sequential(
			self.conv_layer1,
			self.max_pool1,
			self.batch_norm1,
			self.relu1,
			self.conv_layer2,
			self.max_pool2,
			self.batch_norm2,
			self.relu2,
			self.conv_layer3,
			self.max_pool3,
			self.batch_norm3,
			self.relu3
		)
		self.dense_layers = nn.Sequential(
			nn.Linear(832, 256),
			nn.Dropout(0.5),
			nn.Linear(256, 41)
		)
	
	def forward(self, input):
		conv_out = self.conv_layers(input.view(input.shape[0], 1, input.shape[1], input.shape[2]))
		conv_out = self.flatten(conv_out)
		dense_out = self.dense_layers(conv_out)
		return dense_out