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
from captum.attr import IntegratedGradients
import torch.nn.functional as F

sys.path.append("..")
from library.util import Util
from library.loader import Loader


class TrainBase:
    def __init__(self, train_config, finetuning=False):
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

        self.finetuning = finetuning
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

        # choose loss function
        if self.finetuning:
            self.loss = self.finetuning_loss(max_score)
        elif "mse" in self.config.loss.lower():
            self.loss = Util.mse_loss(max_score)
        elif "entropy" in self.config.loss.lower():
            RuntimeError("Unimplemented")
        elif "attention" in self.config.loss.lower():
            self.loss = Util.attention_loss(max_score, anot_lambda=1.0)
        else:
            raise RuntimeError("Invalid loss definition")

    def set_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)

    def set_model(self):
        self.model = Util.select_model(self.model_config, self.config)
        if torch.cuda.is_available():
            self.model.cuda()
        if self.config.parallel:
            self.model = torch.nn.DataParallel(self.model)

    def log_wandb(self, phase, loss, commit=True):
        """
        :param phase: str, "train" or "valid"
        :param loss: float, loss.item()
        :param commit: bool, True if the end of epoch
        :return: None
        """
        wandb.log({"{}_loss".format(phase): loss}, commit=commit)

    def load_dataset(self):
        train_dataset = Util.load_dataset(self.config, "train", finetuning=self.finetuning)
        valid_dataset = Util.load_dataset(self.config, "valid", finetuning=self.finetuning)
        return train_dataset, valid_dataset

    def to_dataloader(self, dataset):
        loader = Loader.to_bert_dataloader if self.prep_type == "bert" else Loader.to_ft_dataloader
        return loader(dataset, self.config.batch_size, self.model_config.attention_hidden_size)

    def predict(self, data_tuple):
        use_attention = (self.config.loss == "attention")
        inputs, _ = data_tuple
        if self.prep_type == "bert":
            output = self.model(inputs[0], inputs[1], inputs[2], attention=use_attention)
        else:
            output = self.model(inputs[0], attention=use_attention)
        if self.config.target_type != "analytic":
            output = output.squeeze(1)
        return output

    def finetuning_loss(self, prompt_score):
        prompt_score = torch.tensor(prompt_score).to(self.device)

        def loss_fn(pred, target, anot, heuristics, inputs):
            pred_score_fixed = pred / prompt_score
            true_score_fixed = target / prompt_score
            loss_first = F.mse_loss(pred_score_fixed, true_score_fixed)

            # heuristics
            loss_second = torch.zeros(loss_first.shape, device=self.device)
            for idx in range(heuristics.shape[0]):
                for jdx in range(heuristics.shape[1]):
                    coef = heuristics[idx][jdx]
                    if coef > 0:
                        gradient = self.int_grad(*inputs, idx=idx, jdx=jdx)
                        target_annotation = anot[idx][jdx] / prompt_score[jdx]
                        loss_second_value = F.mse_loss(gradient, target_annotation)
                        loss_second += loss_second_value.detach()

            loss = loss_first + 10 * loss_second
            return loss

        return loss_fn

    def calc_loss(self, prediction, data_tuple):
        inputs, scores = data_tuple
        if self.finetuning:
            target_score = scores[0] if self.config.target_type == "analytic" else scores[1]
            annotation, heuristics = scores[2], scores[3]
            return self.loss(pred=prediction, target=target_score, anot=annotation, heuristics=heuristics, inputs=inputs)
        else:
            target_score = scores[0] if self.config.target_type == "analytic" else scores[1]
            target_data = (target_score, scores[2]) if self.config.loss == "attention" else target_score
            return self.loss(input=prediction, target=target_data)

    def int_grad(self, input_ids, token_type_ids, attention_mask, idx, jdx):
        arg = (token_type_ids[idx].unsqueeze(0), attention_mask[idx].unsqueeze(0), False, True)
        device = "cuda"
        input_emb = self.model.module.bert.embeddings(input_ids[idx].unsqueeze(0))
        baseline_emb = torch.zeros(input_emb.shape, device=device)

        saliency = IntegratedGradients(self.model, multiply_by_inputs=True)
        grad = saliency.attribute(input_emb, baselines=baseline_emb, target=jdx,
                                  additional_forward_args=arg, n_steps=512, internal_batch_size=128)
        return torch.sum(grad, dim=2).squeeze(0)

    def training_phase(self, train_loader):
        self.model.train()
        losses = []
        for data_tuple in tqdm(train_loader):
            self.optimizer.zero_grad()
            output = self.predict(data_tuple)
            loss = self.calc_loss(output, data_tuple)
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
            loss = self.calc_loss(output, data_tuple)
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
        Util.save_model(self.model, self.config)

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

    def finetune(self):
        self.initialize()
        train_dataset, valid_dataset = self.load_dataset()

        loader = Loader.to_finetuning_dataloader
        ahs = self.model_config.attention_hidden_size
        train_loader = loader(train_dataset, self.config.batch_size, ahs, heuristic=self.config.heuristics)
        valid_loader = loader(valid_dataset, self.config.batch_size, ahs, heuristic=self.config.heuristics)

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


class TrainStatic:
    @staticmethod
    def sweep(train_config, trainer=TrainBase):
        sweep_config = Util.load_sweep_config(train_config.sweep_config_path)
        sweep_id = wandb.sweep(sweep_config, entity=train_config.wandb_entity, project=train_config.wandb_project_name)
        wandb.agent(sweep_id, trainer(train_config), count=train_config.sweep_count)

    @staticmethod
    def cross_validation(train_config, trainer=TrainBase, k=5):
        for idx in range(5):
            train_config.validation = True
            train_config.validation_idx = idx
            trainer(train_config)()
