import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

from dataset import FreesoundDataset
from dataset_2 import FGPA_Dataset
from models.model_1 import Model_1
from models.baseline import Baseline
from models.model_2 import Model_2
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running training on [{device}]")


class LabelTransformer(LabelEncoder):
	def inverse(self, y):
		try:
			return super(LabelTransformer, self).inverse_transform(y)
		except:
			return super(LabelTransformer, self).inverse_transform([y])

	def transform(self, y):
		try:
			return super(LabelTransformer, self).transform(y)
		except:
			return super(LabelTransformer, self).transform([y])

def get_accuracy(model, loader):
	model.eval()
	num_samples = 0
	num_correct = 0
	with torch.no_grad():
		for i, (x_batch, label_batch) in enumerate(loader):
			x_batch, label_batch = x_batch.to(device), label_batch.to(device)

			if args.use_mfcc == True:
				pass
			else:
				x_batch = x_batch.view(x_batch.shape[0], 1, -1)

			y_pred = model(x_batch.float())

			y_pred.to(device)
			preds = y_pred.data.max(1)[1]
			num_samples += preds.size(0)
			num_correct += (preds == label_batch.long()).sum()
		
	return num_correct.item() / num_samples

def train(
	model: nn.Module,
	epochs: int,
	training_dataset: Dataset,
	validation_dataset: Dataset,
	print_every = Config.PRINT_EVERY
) -> nn.Module:
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
	
	train_loss = np.array([])
	validation_loss = np.array([])

	model.to(device)
	optimizer = torch.optim.Adam(model.parameters())
	loss_criterion = nn.CrossEntropyLoss()

	for epoch in range(epochs):
		print(f"--->Running epoch [{epoch}]")
		# Clear loss for this epoch
		running_loss = 0.0

		# Test phase for each epoch
		for phase in phases:
			print(f"Current phase: [{phase}]")

			if phase == "train":
				model.train()
			else:
				model.eval()

			current_dataloader = DataLoader(datasets[phase], batch_size=Config.BATCH_SIZE, shuffle=True)

			for i, data in enumerate(current_dataloader):
				x_batch, label_batch = data
				x_batch, label_batch = x_batch.to(device), label_batch.to(device)

				if args.use_mfcc == True:
					pass
				else:
					x_batch = x_batch.view(x_batch.shape[0], 1, -1)

				optimizer.zero_grad()

				out = model(x_batch.float())
				loss = loss_criterion(out.float(), label_batch.long())

				if phase == 'train':
					# Compute the new gradients
					loss.backward()
					# Update the weights
					optimizer.step()

				running_loss += loss.item()

			
			del current_dataloader

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

			if phase == 'train':
				train_loss = np.append(train_loss, running_loss / (i+1))
			else:
				validation_loss = np.append(validation_loss, current_loss)

			running_loss = 0.0
				
		if epoch % print_every == 0:
			print('Epoch', epoch, '| Validation Accuracy:', get_accuracy(model, DataLoader(validation_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)), '| Train Accuracy:', get_accuracy(model, DataLoader(training_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)))
		print("------------------------------\n\n")

	print('Final epoch', '| Validation Accuracy:', get_accuracy(model, DataLoader(validation_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)), '| Train Accuracy:', get_accuracy(model, DataLoader(training_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)))

	plt.figure(figsize=(8,6))
	plt.plot(train_loss, c='b')
	plt.plot(validation_loss, c='r')
	plt.xticks(np.arange(len(train_loss)))
	plt.legend(['Training loss', 'Validation loss'])
	plt.show()

	model.load_state_dict(torch.load('{}.pt'.format(model.name), map_location=device))
	return model

def main(args) -> None:
	model = None
	if args.model == "baseline":
		model = Baseline(training=True, inplace=True)
	elif args.model == "model_1":
		model = Model_1()
	elif args.model == "model_2":
		model = Model_2()

	assert model is not None

	train_csv = pd.read_csv("../../../Storage/FSDKaggle2018_2/train.csv")

	train_csv = train_csv[train_csv.manually_verified != 0]
	# train_csv = train_csv.iloc[np.random.randint(low=len(train_csv), size=1000)] # Use this for short experiments (only 1000 samples)

	train_filenames = train_csv['fname'].values
	train_labels = train_csv['label'].values
	
	label_transformer = LabelTransformer()
	label_transformer = label_transformer.fit(train_labels)
	np.save('classes.npy', label_transformer.classes_)
	train_label_ids = label_transformer.transform(train_labels)

	# train_idx, validation_idx = next(iter(StratifiedKFold(n_splits=5, shuffle=True, random_state=20180629).split(np.zeros_like(train_label_ids), train_label_ids)))
	# train_files = train_filenames[train_idx]
	# train_labels = train_label_ids[train_idx]
	# val_files = train_filenames[validation_idx]
	# val_labels = train_label_ids[validation_idx]
	# print(f"Number of training samples: {train_files.shape[0]}")
	# print(f"Number of validation samples: {val_files.shape[0]}")

	# model = train(
	# 	model,
	# 	args.epochs,
	# 	FGPA_Dataset("../../../Storage/FSDKaggle2018_2/audio_train/", train_files, train_labels, use_mfcc=args.use_mfcc),
	# 	FGPA_Dataset("../../../Storage/FSDKaggle2018_2/audio_train/", val_files, val_labels, use_mfcc=args.use_mfcc),
	# )


	skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20180629).split(np.zeros_like(train_label_ids), train_label_ids)

	for i, (train_idx, validation_idx) in enumerate(skf):
		train_files = train_filenames[train_idx]
		train_labels = train_label_ids[train_idx]
		val_files = train_filenames[validation_idx]
		val_labels = train_label_ids[validation_idx]
		print("####################################################")
		print(f"Number of training samples: {train_files.shape[0]}")
		print(f"Number of validation samples: {val_files.shape[0]}")

		model = train(
			model,
			args.epochs,
			FGPA_Dataset("../../../Storage/FSDKaggle2018_2/audio_train/", train_files, train_labels, use_mfcc=args.use_mfcc),
			FGPA_Dataset("../../../Storage/FSDKaggle2018_2/audio_train/", val_files, val_labels, use_mfcc=args.use_mfcc),
		)

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Training")
	parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train for.")
	parser.add_argument("-m", "--model", type=str, default="baseline", help="Model to be used for training session.")
	parser.add_argument("-u", "--use_mfcc", type=bool, default=False, help="use_mfcc.")
	args = parser.parse_args()
	main(args)
