import numpy as np
import pandas as pd
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import FreesoundDataset

def load_labels() -> pd.DataFrame:
	train = pd.read_csv('../input/train.csv')
	test = pd.read_csv('../input/sample_submission.csv')


def main(args) -> None:
	pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Training")
	parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs to train for.")
	args = parser.parse_args()
	main(args)