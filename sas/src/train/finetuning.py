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
        self.config.learning_rate = 5e-6

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

    def load_dataset(self):
        train_dataset = Util.load_dataset(self.config, "train", finetuning=True)
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
        term_idx = ord(data_rows["term"]) - 65
        annotation = self.tensor(data_rows["annotation_matrix"][term_idx])
        return data_tuple, gold, term_idx, annotation

    def loss_wrapper(self, prediction, gold, annotation, term_idx):
        attention = prediction[1][:, term_idx, :].unsqueeze(1)
        annotation = annotation.unsqueeze(1)
        if self.config.loss == "attention":
            return self.loss(prediction=prediction[0], gold=gold, attention=attention,
                             annotation=annotation, term_idx=term_idx)
        elif self.config.loss == "gradient":
            return self.loss(prediction=prediction[0], gold=gold, gradient=prediction[1],
                             annotation=annotation, term_idx=term_idx)
        elif self.config.loss == "combination":
            return self.loss(prediction=prediction[0], gold=gold, attention=prediction[1],
                             gradient=prediction[2], annotation=annotation, term_idx=term_idx)
        else:
            RuntimeError("Invalid loss")

    def training_phase(self, train_dataset):
        self.model.train()
        losses = []

        for idx, data_rows in enumerate(train_dataset.iterrows()):
            # filter
            if data_rows[1]["heuristics"] != self.config.heuristics or data_rows[1]["term"] != self.config.term:
                continue

            # training
            self.optimizer.zero_grad()
            input_tuple, gold, term_idx, annotation = self.extract_data(data_rows[1])
            prediction_tuple = self.prediction(input_tuple)
            loss = self.loss_wrapper(prediction_tuple, gold, annotation, term_idx)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        if self.config.wandb:
            self.log_wandb("train", np.mean(losses), commit=True)

        return np.mean(losses)

    def save_model(self):
        Util.save_model(self.model, self.config, finetuning=True)

    def is_best(self, valid_loss):
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            return True
        else:
            return False

    def finetune(self):
        self.initialize()
        train_dataset = self.load_dataset()

        # epoch = self.config.epoch
        for n in range(self.config.epoch):
            epoch = n + 1
            print("epoch:{}".format(epoch))
            train_loss = self.training_phase(train_dataset)
            print("train loss: {:.5f}".format(train_loss))
            if self.is_best(train_loss) or epoch == 1:
                self.save_model()