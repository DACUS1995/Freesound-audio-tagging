import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from models.model_1 import Model_1
from models.model_2 import Model_2
from models.baseline import Baseline
from dataset_2 import FGPA_Dataset
from utils import plot_confusion_matrix, save_results_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating using: [{device}]")


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

def predict(model, test_dataset, evaluation, use_mfcc) -> Tuple:
	# Set the model to evaluation mode
	model.to(device)
	model = model.eval()
	
	# Wrap with no grad because the operations here
	# should not affect the gradient computations
	with torch.no_grad():
		
		predictions = np.array([])
		actual  = np.array([])
		correct = np.array([])
		top = np.empty((0,3))

		batch_size = 1 if evaluation == True else 1
		
		for i, data in enumerate(DataLoader(test_dataset, batch_size)):
			
			# Load the batch
			x_batch, label_batch = data
			x_batch, label_batch = x_batch.to(device), label_batch.to(device)
				
			if use_mfcc == True:
				pass
			else:
				x_batch = x_batch.view(x_batch.shape[0], 1, -1)

			# Get the output of the model
			out = model(x_batch.float())
			out_1 = out.max(1)[1]

			out = out.cpu().detach().numpy()
			top_3 = out.reshape(-1).argsort()[-3:][::-1]
			top_3 = top_3.reshape((1, -1))
			top = np.append(top, top_3, axis=0)

			actual = np.append(actual, label_batch.cpu().detach().numpy())
			correct = np.append(correct, (out_1 == label_batch.long()).cpu().detach().numpy())
			predictions = np.append(predictions, out_1.cpu().detach().numpy())
							
	return predictions, correct, actual, top


def main(args):
	if args.model == "baseline":
		model = Baseline(inplace=True, training=False)
	elif args.model == "model_1":
		model = Model_1()
	elif args.model == "model_2":
		model = Model_2()
	
	model.load_state_dict(torch.load('{}.pt'.format(model.name), map_location=device))

	test_csv = pd.read_csv("../../../Storage/FSDKaggle2018_2/train.csv")

	if args.evaluation == True:
		test_csv = test_csv[test_csv.manually_verified == 0]

	test_filenames = test_csv['fname'].values
	test_labels = test_csv['label'].values

	print(f"Number of testing samples: {test_filenames.shape[0]}")

	label_transformer = LabelTransformer()

	if os.path.isfile("./" + Config.CLASSES_FILE):
		label_transformer.classes_ = np.load(Config.CLASSES_FILE)
	else:
		train_csv = pd.read_csv("../../../Storage/FSDKaggle2018_2/train.csv")
		train_labels = train_csv['label'].values
		label_transformer = label_transformer.fit(train_labels)

	if args.evaluation == True:
		test_label_ids = label_transformer.transform(test_labels)
	else:
		test_label_ids = np.zeros(test_labels.shape)

	test_dataset = FGPA_Dataset("../../../Storage/FSDKaggle2018_2/audio_train/", test_filenames, test_label_ids, use_mfcc=True)
	predictions, correct, actual, top = predict(model, test_dataset, args.evaluation, args.use_mfcc)

	df = pd.DataFrame(data={
		"fname": test_filenames,
		"label": predictions,
		"correct": correct
	})

	df.to_csv("verified.csv", index=False)

	if args.evaluation == False:
		# Compute confusion matrix
		cnf_matrix = confusion_matrix(actual, predictions)
		np.set_printoptions(precision=2)
	
		# Plot non-normalized confusion matrix
		plt.figure(figsize=(16,16))
		plot_confusion_matrix(
			cnf_matrix,
			classes=label_transformer.classes_,
			title='Title'
		)

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Testing")
	parser.add_argument("-m", "--model", type=str, default="baseline", help="Model to be used for testing session.")
	parser.add_argument("-e", "--evaluation", type=bool, default=False, help="The current run is for contest evaluation.")
	parser.add_argument("-s", "--save", type=bool, default=False, help="Save the resulted labels in a csv file.")
	parser.add_argument("-u", "--use_mfcc", type=bool, default=False, help="use_mfcc.")
	args = parser.parse_args()
	main(args)