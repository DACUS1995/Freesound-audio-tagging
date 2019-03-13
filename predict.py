import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from models.baseline import Baseline
from dataset_2 import FGPA_Dataset
from utils import plot_confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def predict(model, test_dataset):
	print("Predicting test dataset.")
	# Set the model to evaluation mode
	model.to(device)
	model = model.eval()
	
	# Wrap with no grad because the operations here
	# should not affect the gradient computations
	with torch.no_grad():
		
		predictions = np.array([])
		actual  = np.array([])
		correct = np.array([])
		
		for i, data in enumerate(DataLoader(test_dataset, Config.BATCH_SIZE)):
			
			# Load the batch
			x_batch, label_batch = data
			# Send to device for faster computations
			x_batch, label_batch = x_batch.to(device), label_batch.to(device)
			x_batch = x_batch.view(x_batch.shape[0], 1, -1)
			
			# Get the output of the model
			out = model(x_batch.float()).max(1)[1]
			# Send to device for faster computations
			out = out.to(device)
			
			actual = np.append(actual, label_batch.cpu().detach().numpy())
			correct = np.append(correct, (out == label_batch.long).cpu().detach().numpy())
			predictions = np.append(predictions, out.cpu().detach().numpy())
							
	return predictions, correct, actual


def main(args):
	if args.model == "baseline":
		model = Baseline()
		model.load_state_dict(torch.load('{}.pt'.format(model.name), map_location=device))
	elif args.model == "model_1":
		pass

	test_csv = pd.read_csv("../../../Storage/FSDKaggle2018_2/test_post_competition.csv")
	test_csv = test_csv[test_csv.usage != 'Ignored']

	test_filenames = test_csv['fname'].values
	test_labels = test_csv['label'].values

	label_transformer = LabelTransformer()
	label_transformer = label_transformer.fit(test_labels)
	train_label_ids = label_transformer.transform(test_labels)
	print(len(np.unique(train_label_ids)))

	test_label_ids = label_transformer.transform(test_labels)

	test_dataset = FGPA_Dataset("../../../Storage/FSDKaggle2018_2/audio_test/", test_filenames, test_label_ids)
	predictions, correct, actual = predict(model, test_dataset)

	# Compute confusion matrix
	cnf_matrix = confusion_matrix(actual, predictions)
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	plt.figure(figsize=(16,16))
	plot_confusion_matrix(
		cnf_matrix, classes=label_transformer.classes_,
		title='Title'
	)

	# Plot normalized confusion matrix
	# plt.figure(figsize=(20,20))
	# plot_confusion_matrix(cnf_matrix, classes=label_transformer.classes_, normalize=True,
	#                       title='Normalized confusion matrix')

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Predict")
	parser.add_argument("-m", "--model", type=str, default="baseline", help="Model to be used for training session.")
	args = parser.parse_args()
	main(args)