import string

import pandas as pd
import torch
import wandb
from transformers import AdamW
from pathlib import Path
from torch.optim import SGD, Adam
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import uuid
import datetime
from captum.attr import IntegratedGradients
import torch.nn.functional as F
from pprint import pprint

sys.path.append("..")
from library.util import Util
from library.loss import Loss
from library.loader import Loader
from train.base import TrainBase

class TrainMasking(TrainBase):
    def __init__(self, train_config, masking_span):
        super().__init__(train_config)
        self.masking_span = masking_span

    def load_dataset(self):
        train_dataset = Util.load_masking_dataset(self.config, "train", masking_span=self.masking_span)
        valid_dataset = Util.load_masking_dataset(self.config, "valid", masking_span=self.masking_span)
        return train_dataset, valid_dataset

    def save_model(self):
        Util.save_masking_model(self.model, self.config, masking_span=self.masking_span)

    def execute(self,):
        self.initialize()
        train_dataset, valid_dataset = self.load_dataset()
        self.train(train_dataset, valid_dataset)
