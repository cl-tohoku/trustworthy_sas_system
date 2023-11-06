import torch
import wandb
from transformers import AdamW
from pathlib import Path
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import uuid
import datetime
import random
from torch.utils.data import Dataset, Subset

from captum.attr import IntegratedGradients
import torch.nn.functional as F
from scipy.special import roots_legendre
from collections import defaultdict

sys.path.append("..")
from library.util import Util
from library.loss import Loss
from library.loader import Loader
from library.attribution import FeatureAttribution as FA


class TrainBase:
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
        self.wandb_name = datetime.datetime.now().strftime('%Y%m%d%H%M')
        # set output size for prompt
        self.model_config.output_size = self.prompt_config.scoring_item_num
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        })
        self.config.update(wandb.config)
        self.model_config.update(wandb.config)

    def set_wandb(self):
        wandb.watch(self.model)

    def set_loss(self):
        # set max score
        max_score = self.prompt_config.max_scores
        # set MSE loss
        self.loss = Loss.mse_loss(max_score)

    def set_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)

    def set_model(self):
        self.model = Util.select_model(self.model_config, self.config)
        if torch.cuda.is_available():
            self.model.cuda()
        if self.config.parallel:
            self.model = torch.nn.DataParallel(self.model)

    def log_wandb(self, phase, loss, commit=True):
        wandb.log({"{}_loss".format(phase): loss}, commit=commit)

    def load_dataset(self):
        prep_name, mode = self.config.preprocess_name, self.config.mode
        if "superficial" in self.config.mode.lower():
            sf_term, sf_idx = self.config.sf_term, self.config.sf_idx
            assert sf_term is not None and sf_idx is not None
            train_dataset = Util.load_sf_dataset(sf_term, sf_idx, prep_name, "train", self.config.dataset_dir)
            valid_dataset = Util.load_sf_dataset(sf_term, sf_idx, prep_name, "valid", self.config.dataset_dir)
        else:
            train_dataset = Util.load_dataset_static(prep_name, "train", mode, self.config.dataset_dir)
            valid_dataset = Util.load_dataset_static(prep_name, "valid", mode, self.config.dataset_dir)

        return train_dataset, valid_dataset

    def to_dataloader(self, dataset):
        loader = Loader.to_bert_dataloader if self.prep_type == "bert" else Loader.to_ft_dataloader
        return loader(dataset, self.config.batch_size, self.model_config.attention_hidden_size)

    def predict(self, data_tuple):
        inputs, _ = data_tuple
        output = self.model(inputs[0], inputs[1], inputs[2], attention=True)
        return output

    def training_phase(self, train_loader):
        self.model.train()
        losses = []
        for data_tuple in tqdm(train_loader):
            self.optimizer.zero_grad()
            output = self.predict(data_tuple)
            loss = self.loss(input=output[0], target=data_tuple[1][0])
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        if self.config.wandb:
            self.log_wandb("train", np.mean(losses), commit=False)
        return np.mean(losses)

    def validation_phase(self, valid_loader):
        self.model.eval()
        losses = []
        for data_tuple in tqdm(valid_loader):
            output = self.predict(data_tuple)
            loss = self.loss(input=output[0], target=data_tuple[1][0])
            losses.append(loss.item())

        if self.config.wandb:
            self.log_wandb("valid", np.mean(losses), commit=True)
        return np.mean(losses)

    def is_best(self, valid_loss):
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            return True
        else:
            return False

    def save_model(self):
        file_name = "{}.state".format(self.config.script_name)
        os.makedirs(self.config.model_dir, exist_ok=True)
        output_path = Path(self.config.model_dir) / file_name
        state_dict = self.model.state_dict()
        if self.config.parallel:
            state_dict = Util.replace_parallel_state_dict(state_dict)
        torch.save(state_dict, output_path)

    def train(self, train_dataset, valid_dataset):
        train_loader = self.to_dataloader(train_dataset)
        valid_loader = self.to_dataloader(valid_dataset)

        for n in range(self.config.epoch):
            epoch = n + 1
            print("epoch:{}".format(epoch))
            train_loss = self.training_phase(train_loader)
            valid_loss = self.validation_phase(valid_loader)
            print("train loss: {:.3f}, valid loss: {:.3f}".format(train_loss, valid_loss))
            if self.is_best(valid_loss) or epoch == 1:
                self.save_model()

    def __call__(self):
        self.initialize()
        train_dataset, valid_dataset = self.load_dataset()
        self.train(train_dataset, valid_dataset)


class TrainSupervising:
    def __init__(self, train_config):
        self.config = train_config
        self.model_config = Util.load_model_config(train_config.model_config_path)
        self.prompt_config = Util.load_prompt(self.config)
        self.model, self.loss, self.mse_loss = None, None, None
        self.optimizer = None
        self.best_valid_loss = 1e+10
        self.prep_type = self.config.preprocessing_type
        # setting group & id
        self.group = "{}_{}".format(self.config.script_name, self.config.model_name)
        self.experiment_name = train_config.unique_id
        self.use_experiment_name = self.experiment_name is not None
        self.wandb_name = datetime.datetime.now().strftime('%Y%m%d%H%M')
        # set output size for prompt
        self.model_config.output_size = self.prompt_config.scoring_item_num
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # A -> 0, B->1
        self.target_idx = ord(self.config.sf_term) - 65
        prev_model_name = "{}_{}-{}.state".format(self.config.prev_script_name, self.config.sf_term, self.config.sf_idx)
        self.model_path = Path(self.config.model_dir) / prev_model_name

        # 同じに設定する
        self.step_size = 16
        self.internal_size = 16

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

        # set grad loss function
        self.loss = Loss.grad_loss(max_score, _lambda=1e-1, _print=True)
        self.mse_loss = Loss.mse_loss(max_score)


    def set_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)

    def set_model(self):
        # load previous model
        self.model = Util.load_model_from_path(self.model_path, self.model_config)

        if torch.cuda.is_available():
            self.model.cuda()
        if self.config.parallel:
            self.model = torch.nn.DataParallel(self.model)

    def log_wandb(self, phase, loss, commit=True):
        wandb.log({"{}_loss".format(phase): loss}, commit=commit)

    def load_dataset(self):
        dataset = Util.load_dataset_static(self.config.preprocess_name, "chosen", self.config.mode,
                                           self.config.dataset_dir)

        def transform(list_data):
            return torch.tensor(list_data, device=self.device).unsqueeze(0)

        baseline_model = Util.load_model_from_path(self.model_path, self.model_config)
        baseline_model.cuda()

        grad_list = []
        for ii, tti, am in tqdm(zip(dataset["input_ids"], dataset["token_type_ids"], dataset["attention_mask"])):
            ii, tti, am = transform(ii), transform(tti), transform(am)
            grad = FA.calc_int_grad(baseline_model, ii, (tti, am), target_idx=self.target_idx, step_size=128,
                                    internal_size=128)
            grad = FA.compress(tensor=grad, summation=False)
            grad_list.append(grad)

        dataset["orig_gradient"] = grad_list
        baseline_model.cpu()

        corr_dataset = Util.load_dataset(self.config, "train", self.config.script_name, "sv")
        sampling_size = 100
        indices = random.sample(range(sampling_size), sampling_size)
        corr_dataset = corr_dataset.iloc[indices].reset_index(drop=True)

        return dataset, corr_dataset

    def to_dataloader(self, dataset):
        loader = Loader.to_sv_dataloader
        return loader(dataset, self.config.batch_size, self.model_config.attention_hidden_size)

    def to_corr_dataloader(self, dataset):
        loader = Loader.to_bert_dataloader if self.prep_type == "bert" else Loader.to_ft_dataloader
        return loader(dataset, self.config.batch_size, self.model_config.attention_hidden_size)

    def calc_gradient(self, data_tuple):
        inputs, scores = data_tuple
        token, args, orig_grad = inputs[0], (inputs[1], inputs[2]), inputs[3]

        _iterator = FA.calc_int_grad_for_training(self.model, token, args, parallel=True, step_size=self.step_size,
                                                  internal_size=self.internal_size, target_idx=self.target_idx)
        for grad, pred, batch, term in _iterator:
            loss = self.loss(pred=pred, gold=data_tuple[1][0], gradient=grad, orig_gradient=orig_grad[batch],
                             annotation=data_tuple[1][2], batch_idx=batch, term_idx=term)
            yield loss

    def predict(self, data_tuple):
        inputs, _ = data_tuple
        output = self.model(inputs[0], inputs[1], inputs[2], attention=True)
        return output

    def training_phase(self, train_loader):
        self.model.train()
        losses = []
        for data_tuple in tqdm(train_loader):
            for loss in self.calc_gradient(data_tuple):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

        if self.config.wandb:
            self.log_wandb("train", np.mean(losses), commit=False)
        return np.mean(losses)

    def correction_phase(self, corr_loader):
        self.model.train()
        losses = []
        for data_tuple in tqdm(corr_loader):
            self.optimizer.zero_grad()
            output = self.predict(data_tuple)
            loss = self.mse_loss(input=output[0], target=data_tuple[1][0])
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return np.mean(losses)

    def save_model(self):
        file_name = "{}.state".format(self.config.script_name)
        os.makedirs(self.config.model_dir, exist_ok=True)
        output_path = Path(self.config.model_dir) / file_name
        state_dict = self.model.state_dict()
        if self.config.parallel:
            state_dict = Util.replace_parallel_state_dict(state_dict)
        torch.save(state_dict, output_path)

    def train(self, train_dataset, corr_dataset):
        train_loader = self.to_dataloader(train_dataset)
        corr_loader = self.to_corr_dataloader(corr_dataset)

        for n in range(self.config.epoch):
            epoch = n + 1
            print("\nepoch:{}".format(epoch))
            train_loss = self.training_phase(train_loader)
            print("\ntrain loss: {:.3f}".format(train_loss))
            corr_loss = self.correction_phase(corr_loader)
            print("\ncorr loss: {:.3f}".format(corr_loss))
            self.save_model()

    def __call__(self):
        self.initialize()
        train_dataset, corr_dataset = self.load_dataset()
        self.train(train_dataset, corr_dataset)