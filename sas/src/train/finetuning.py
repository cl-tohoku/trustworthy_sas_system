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


class TrainFinetuning:
    def __init__(self, train_config):
        self.config = train_config
        self.model_config = Util.load_model_config(train_config.model_config_path)
        self.prompt_config = Util.load_prompt(self.config)
        self.model, self.loss = None, None
        self.optimizer = None
        self.best_valid_loss = 1e+10
        self.prep_type = self.config.preprocessing_type
        # setting group & id
        self.group = "{}_{}".format(self.config.script_name, self.config.model_name)
        self.experiment_name = train_config.unique_id
        self.use_experiment_name = self.experiment_name is not None
        self.wandb_name = datetime.datetime.now().strftime('%Y%m%d%H%M')
        # set output size for prompt
        if self.config.target_type == "analytic":
            self.model_config.output_size = self.prompt_config.scoring_item_num
        else:
            self.model_config.output_size = 1

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config.learning_rate = 1e-6

    def initialize(self,):
        if self.config.wandb:
            self.init_wandb()
        self.set_model()
        self.set_loss()
        self.set_optimizer()
        if self.config.wandb:
            self.set_wandb()

    def init_wandb(self):
        self.wandb_name = str(uuid.uuid4())[:8]
        self.best_valid_loss = 1e+10
        wandb.init(project=self.config.wandb_project_name, entity=self.config.wandb_entity,
                   group=self.group, name=self.wandb_name)
        wandb.config.update({
            "epoch": self.config.epoch,
            "learning_rate": self.config.learning_rate,
            "loss": self.config.loss,
            "target_type": self.config.target_type,
        })
        self.config.update(wandb.config)
        self.model_config.update(wandb.config)

    def set_wandb(self):
        wandb.watch(self.model)

    def set_loss(self):
        # set max score
        if self.config.target_type == "analytic":
            max_score = self.prompt_config.max_scores
        else:
            max_score = sum(self.prompt_config.max_scores)

        if "attention" in self.config.loss.lower():
            self.loss = Loss.attn_loss(max_score)
        elif "gradient" in self.config.loss.lower():
            self.loss = Loss.grad_loss(max_score)
        elif "combination" in self.config.loss.lower():
            self.loss = Loss.comb_loss(max_score)
        else:
            raise RuntimeError("Invalid loss definition")

    def set_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-2)

    def set_model(self, ):
        self.model = Util.load_model(self.config, self.model_config)
        # self.model = Util.select_model(self.model_config, self.config)
        [p.retain_grad() for p in self.model.parameters()]
        if torch.cuda.is_available():
            self.model.cuda()
        if self.config.parallel:
            self.model = torch.nn.DataParallel(self.model)

    def log_wandb(self, phase, loss, commit=True):
        wandb.log({"{}_loss".format(phase): loss}, commit=commit)

    def load_dataset(self, term, cluster_size, selection_size):
        train_dataset = Util.load_finetuning_dataset(self.config, "train", term, cluster_size, selection_size)
        return train_dataset

    def calc_gradient(self, inputs):
        embedding_func = self.model.module.bert.embeddings
        grad_list = [Util.calc_gradient_static(self.model, embedding_func, inputs[0], inputs[1], inputs[2], target=idx)
                     for idx in range(self.model_config.output_size)]
        return torch.stack(grad_list).permute(1, 0, 2)

    def prediction(self, inputs):
        output = self.model(input_ids=inputs[0], token_type_ids=inputs[1], attention_mask=inputs[2], attention=True)
        if self.config.loss.lower() == "gradient":
            grad = self.calc_gradient(inputs)
            output = (output[0], grad)
        elif self.config.loss.lower() == "combination":
            grad = self.calc_gradient(inputs)
            output = output + (grad, )
        return output

    def tensor(self, array):
        return torch.tensor(array, device=self.device).unsqueeze(0)

    def extract_data(self, data_rows):
        ii, tti, am = data_rows["input_ids"], data_rows["token_type_ids"], data_rows["attention_mask"]
        ii, tti, am = self.tensor(ii), self.tensor(tti), self.tensor(am)
        data_tuple = (ii, tti, am)
        gold = self.tensor(data_rows["score_vector"])
        annotation = self.tensor(data_rows["annotation_matrix"])
        use_heuristics = bool(data_rows["Heuristics"])
        return data_tuple, gold, annotation, use_heuristics

    def loss_wrapper(self, prediction, gold, annotation, term_idx, use_heuristics):
        attention = prediction[1]
        annotation = annotation
        if self.config.loss == "attention":
            return self.loss(prediction=prediction[0], gold=gold, attention=attention,
                             annotation=annotation, term_idx=term_idx, use_attention=use_heuristics)
        elif self.config.loss == "gradient":
            return self.loss(prediction=prediction[0], gold=gold, gradient=prediction[1],
                             annotation=annotation, term_idx=term_idx)
        elif self.config.loss == "combination":
            return self.loss(prediction=prediction[0], gold=gold, attention=prediction[1],
                             gradient=prediction[2], annotation=annotation, term_idx=term_idx)
        else:
            RuntimeError("Invalid loss")

    def choice_dataset(self, dataset_df):
        heuristics_size = (dataset_df["Heuristics"] == True).sum()
        heuristics_df = dataset_df[dataset_df["Heuristics"] == True]
        normal_df = dataset_df[dataset_df["Heuristics"] == False].sample(n=heuristics_size, replace=False)
        return pd.concat([heuristics_df, normal_df]).sample(frac=1, random_state=42)

    def training_phase(self, train_dataset, term):
        self.model.train()
        losses = []
        term_idx = ord(term) - 65

        for idx, data_rows in enumerate(train_dataset.iterrows()):
            # training
            self.optimizer.zero_grad()
            input_tuple, gold, annotation, use_heuristics = self.extract_data(data_rows[1])
            prediction_tuple = self.prediction(input_tuple)
            loss = self.loss_wrapper(prediction_tuple, gold, annotation, term_idx, use_heuristics)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        if self.config.wandb:
            self.log_wandb("train", np.mean(losses), commit=True)

        return np.mean(losses)

    def save_model(self, term, cluster_size, selection_size):
        Util.save_finetuning_model(self.model, self.config, term, cluster_size, selection_size)

    def is_best(self, valid_loss):
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            return True
        else:
            return False

    def finetune(self, term, cluster_size, selection_size):
        self.initialize()
        print("term: {}, c_size: {}, s_size: {}".format(term, cluster_size, selection_size))
        train_dataset = self.load_dataset(term, cluster_size, selection_size)

        # epoch = self.config.epoch
        pprint(self.config)
        used_sample_size = 0
        while used_sample_size < 10000:
            choice_df = self.choice_dataset(train_dataset)
            print("used sample size:{}".format(used_sample_size))
            train_loss = self.training_phase(choice_df, term)
            print("train loss: {:.5f}".format(train_loss))
            if self.is_best(train_loss) or used_sample_size == 0:
                self.save_model(term, cluster_size, selection_size)
            used_sample_size += len(choice_df)

    def execute(self):
        term_list = list(string.ascii_uppercase)[:self.model_config.output_size]
        cluster_list, selection_list = list(range(10, 11)), list(range(3, 11))

        from itertools import product
        for term, cluster_size, selection_size in product(term_list, cluster_list, selection_list):
            self.finetune(term, cluster_size, selection_size)
