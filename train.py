import numpy as np
import pandas as pd
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from dataset import FreesoundDataset
from models.baseline import Baseline
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running traing on [{device}]")

def load_labels() -> Tuple:
	train = pd.read_csv('../input/train.csv')
	test = pd.read_csv('../input/sample_submission.csv')
	return train, test


def train(
	model: nn.Module,
	epochs: int,
	training_dataset: Dataset,
	validation_dataset: Dataset
) -> None:
	print("-->Begining training")

	if validation_dataset is not None:
		datasets = {
			'train': training_dataset,
			'validation': validation_dataset
		}
		previous_loss = 100
		phases = ['train', 'validation']
	else:
		train_dataset = training_dataset
		datasets = {
			'train': train_dataset,
			'validation': None
		}
		phases = ['train']

	model.to(device)
	optimizer = torch.optim.Adam(model.parameters())
	criterion = nn.CrossEntropyLoss()

	for epoch in range(epochs):
		print(f"--->Running epoch [{epoch}]")
		# Clear loss for this epoch
		running_loss = 0.0

		for phase in phases:
			print(f"Current phase: [{phase}]")

			if phase == "train":
				model.train()
			else:
				model.eval()

			for i, (data, label) in enumerate(DataLoader(datasets[phase], batch_size=Config.BATCH_SIZE, shuffle=True)):
				data.to(device)
				x_batch, label_batch = data


				# optimizer.zero_grad()

				# out = model(x_batch)
				# loss = criterion(out, label_batch)

				if phase == 'train':
					# Compute the new gradients
					loss.backward()
					# Update the weights
					optimizer.step()
				running_loss += loss.item()


			print('{} loss: {}'.format(phase, running_loss / (i+1)))
			if phase == 'validation':
				current_loss = running_loss / (i+1)
				if current_loss < previous_loss:
					print('Loss decreased. Saving the model..')
					# If loss decreases,
					# save the current model as the best-shot checkpoint
					torch.save(model.state_dict(), '{}.pt'.format(model.name))

					# update the value of the loss
					previous_loss = current_loss

			if phase=='train':
				train_loss = np.append(train_loss, running_loss / (i+1))
			else:
				validation_loss = np.append(validation_loss, current_loss)

def main(args) -> None:
	model = None
	if args.model == "baseline":
		model = Baseline()
	elif args.model == "model_1":
		pass

	train(
		model,
		args.epochs,
		FreesoundDataset(mode="train"),

	)

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Training")
	parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs to train for.")
	parser.add_argument("-m", "--model", type=str, default="baseline", help="Model to be used for training session.")
	args = parser.parse_args()
	main(args)