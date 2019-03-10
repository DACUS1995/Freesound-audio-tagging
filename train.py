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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running traing on [{device}]")

def load_labels() -> Tuple:
	train = pd.read_csv('../input/train.csv')
	test = pd.read_csv('../input/sample_submission.csv')
	return train, test


def train(
	model: nn.Module,
	epochs: int,
	dataset: Dataset
) -> None:
	print("-->Begining training")

	model.to(device)
	model.train()
	optimizer = torch.optim.Adam(model.parameters())
	criterion = nn.CrossEntropyLoss()

	for epoch in range(epochs):
		print(f"--->Running epoch [{epoch}]")

		# TODO problem when using batch_size > 1, samples must have the same size
		for i, (data, label) in enumerate(DataLoader(dataset, batch_size=1, shuffle=True)):
			print(label)
			data.to(device)

			# optimizer.zero_grad()

			# out = model(data)
			# loss = criterion(out, label)

			# loss.backward()
			# optimizer.step()
		

def main(args) -> None:
	model = None
	if args.model == "baseline":
		model = Baseline()
	elif args.model == "model_1":
		pass

	train(
		model,
		args.epochs,
		FreesoundDataset()
	)

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Training")
	parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs to train for.")
	parser.add_argument("-m", "--model", type=str, default="baseline", help="Model to be used for training session.")
	args = parser.parse_args()
	main(args)