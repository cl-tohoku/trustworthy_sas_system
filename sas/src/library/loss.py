import copy
import os
import yaml
import sys
from pathlib import Path
import pickle
from collections import OrderedDict
import torch
from dataclasses import dataclass, asdict
import torch.nn.functional as F
import pandas as pd

sys.path.append("..")
from library.structure import PreprocessConfig, TrainConfig, ModelConfig, EvalConfig, PromptConfig
from model.bert import SimpleBert
from model.ffnn import FFNN
from model.lstm import LSTM


class Loss:
    def __init__(self):
        pass

    @staticmethod
    def mse_loss(prompt_score):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prompt_score = torch.tensor(prompt_score).to(device)

        def loss_fn(input, target):
            pred_score_fixed = input / prompt_score
            true_score_fixed = target / prompt_score
            loss = F.mse_loss(pred_score_fixed, true_score_fixed)
            return loss

        return loss_fn


    @staticmethod
    def grad_loss(prompt_score, _lambda=1.0, _print=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prompt_score = torch.tensor(prompt_score).to(device)

        def loss_fn(pred, gold, gradient, orig_gradient, annotation, term_idx=None, batch_idx=None):
            # ordinal mse loss
            pred_score_fixed = pred / prompt_score
            true_score_fixed = gold / prompt_score
            loss_first = F.mse_loss(pred_score_fixed, true_score_fixed)

            # calc grad term
            transformed_annotation = annotation[batch_idx][term_idx]
            transformed_annotation = transformed_annotation.unsqueeze(1).repeat(1, gradient.shape[1])
            ideal_gradient = (transformed_annotation * orig_gradient).detach()

            a, b = torch.flatten(torch.sum(gradient, dim=1)), torch.flatten(torch.sum(ideal_gradient, dim=1))
            loss_second = 1.0 - torch.abs(F.cosine_similarity(a, b, dim=0))

            # 勾配が0のときは学習しない
            if torch.sum(gradient) != 0.00:
                loss = loss_first + _lambda * loss_second
                message = 'Loss:{:.4f}, Loss_1:{:.4f}, Loss_2:{:.8f}'
                if _print:
                    print(message.format(loss, loss_first, _lambda * loss_second))
            else:
                loss = loss_first
                message = 'Loss:{:.4f}, Loss_1:{:.4f}'
                if _print:
                    print(message.format(loss, loss_first))

            return loss

        return loss_fn


    @staticmethod
    def attn_loss(prompt_score, _lambda=1.0):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prompt_score = torch.tensor(prompt_score).to(device)

        def loss_fn(prediction, gold, attention, annotation, term_idx=-1, use_attention=True):
            # ordinal mse loss
            pred_score_fixed = prediction / prompt_score
            true_score_fixed = gold / prompt_score
            loss_first = F.mse_loss(pred_score_fixed, true_score_fixed)

            # attention loss
            if use_attention:
                selected_attn = attention
                fixed_annotation = torch.softmax((annotation - 1.0) * 1e+10, dim=2)
                if term_idx == -1:
                    loss_second = F.mse_loss(selected_attn, fixed_annotation)
                else:
                    loss_second = F.mse_loss(selected_attn[:, term_idx, :], fixed_annotation[:, term_idx, :])
            else:
                loss_second = 0.0

            loss = loss_first + _lambda * loss_second

            print('\rLoss:{:.5f}, Loss_1:{:.5f}, Loss_2:{:.10f}, '.format(loss, loss_first, loss_second, ), end='')
            return loss

        return loss_fn