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

        self.loss = self.attn_loss(max_score)

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

    def grad_loss(self, prompt_score):
        prompt_score = torch.tensor(prompt_score).to(self.device)

        def loss_fn(prediction, gold, grad, annotation, term_idx):
            # ordinal mse loss
            pred_score_fixed = prediction / prompt_score
            true_score_fixed = gold/ prompt_score
            loss_first = F.mse_loss(pred_score_fixed, true_score_fixed)

            grad_zero = (1.0 - annotation.to(torch.float32).squeeze(0)) * grad.squeeze(0)
            loss_second = torch.norm(grad_zero) * 1e+10
            # loss = loss_first + 1.0 * loss_second
            loss = loss_first

            print('\rLoss:{:.5f}, Loss_1:{:.5f}, Loss_2:{:.5f}'.format(loss, loss_first, loss_second,), end='')
            return loss

        return loss_fn

    def attn_loss(self, prompt_score):
        prompt_score = torch.tensor(prompt_score).to(self.device)

        def loss_fn(prediction, gold, attention, annotation, term_idx):
            # ordinal mse loss
            pred_score_fixed = prediction / prompt_score
            true_score_fixed = gold / prompt_score
            loss_first = F.mse_loss(pred_score_fixed, true_score_fixed)

            # attntion loss, attention mse
            selected_attn = attention[term_idx].squeeze(0)
            annotation_sum = torch.sum(annotation)
            if annotation_sum < 1:
                fixed_anno = torch.ones(annotation.shape[1], device=self.device) / annotation.shape[1]
            else:
                fixed_anno = annotation.squeeze(0) / annotation_sum
            loss_second = F.mse_loss(selected_attn, fixed_anno)
            loss = loss_first + 10.0 * loss_second

            print('\rLoss:{:.5f}, Loss_1:{:.5f}, Loss_2:{:.10f}, '.format(loss, loss_first, loss_second,), end='')
            # print('\rLoss_2:{:.10f}, '.format(loss_second,), end='')
            return loss

        return loss_fn

    def calc_int_grad(self, input_ids, token_type_ids, attention_mask, target):
        self.model.eval()
        arg = (token_type_ids, attention_mask, False, True)
        device = "cuda"
        input_emb = self.model.module.bert.embeddings(input_ids)
        baseline_emb = torch.zeros(input_emb.shape, device=device)

        saliency = IntegratedGradients(self.model, multiply_by_inputs=True)
        grad = saliency.attribute(input_emb, baselines=baseline_emb, target=target,
                                  additional_forward_args=arg, n_steps=512, internal_batch_size=128)
        self.model.train()
        return torch.sum(grad, dim=2).to(torch.float32)

    def prediction_with_grad(self, input_ids, token_type_ids, attention_mask, term_idx):
        input_emb = self.model.module.bert.embeddings(input_ids)
        prediction = self.model(input_ids=input_emb, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                inputs_embeds=True)
        grad = torch.autograd.grad(outputs=prediction[0][term_idx], inputs=input_emb, retain_graph=True)[0]
        return prediction, torch.sum(grad, dim=2).to(torch.float32)

    def prediction_attn(self, input_ids, token_type_ids, attention_mask, term_idx):
        input_emb = self.model.module.bert.embeddings(input_ids)
        prediction, attention = self.model(input_ids=input_emb, token_type_ids=token_type_ids,
                                           attention_mask=attention_mask, inputs_embeds=True, attention=True)
        return prediction, attention

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

    def training_phase(self, train_dataset):
        self.model.train()
        losses = []
        dataset_size = len(train_dataset)
        for idx, data_rows in enumerate(train_dataset.iterrows()):
            if data_rows[1]["heuristics"] != self.config.heuristics or data_rows[1]["term"] != self.config.term:
                continue
            self.optimizer.zero_grad()
            data_tuple, gold, term_idx, annotation = self.extract_data(data_rows[1])
            prediction, attention = self.prediction_attn(data_tuple[0], data_tuple[1], data_tuple[2], term_idx)
            loss = self.loss(prediction, gold, attention, annotation, term_idx)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        if self.config.wandb:
            self.log_wandb("train", np.mean(losses), commit=False)
        return np.mean(losses)

    def save_model(self):
        Util.save_model(self.model, self.config, heuristics=True)

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
        epoch = 50
        for n in range(epoch):
            epoch = n + 1
            print("epoch:{}".format(epoch))
            train_loss = self.training_phase(train_dataset)
            print("train loss: {:.5f}".format(train_loss))
            if self.is_best(train_loss) or epoch == 1:
                self.save_model()