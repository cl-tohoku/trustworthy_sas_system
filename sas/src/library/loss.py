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
    def grad_loss(prompt_score, _lambda=1.0):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prompt_score = torch.tensor(prompt_score).to(device)

        def loss_fn(prediction, gold, gradient, annotation, term_idx=None):
            # ordinal mse loss
            pred_score_fixed = prediction / prompt_score
            true_score_fixed = gold / prompt_score
            loss_first = F.mse_loss(pred_score_fixed, true_score_fixed)

            # calc grad term
            transformed_annotation = 1.0 - annotation
            transformed_annotation = transformed_annotation.unsqueeze(3).repeat(1, 1, 1, gradient.shape[3])
            ideal_gradient = transformed_annotation * gradient
            # loss_second = F.mse_loss(gradient, ideal_gradient)
            a, b = torch.flatten(gradient), torch.flatten(ideal_gradient)
            loss_second = 1.0 - torch.abs(F.cosine_similarity(a, b, dim=0))

            loss = loss_first + _lambda * loss_second

            grad_sample = gradient[0][0][0][0]
            message = '\rLoss:{:.5f}, Loss_1:{:.5f}, Loss_2:{:.5f}, grad:{}'
            print(message.format(loss, loss_first, _lambda * loss_second, grad_sample), end='')
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

    @staticmethod
    def comb_loss(prompt_score, _lambda_attn=1.0, _lambda_grad=1.0):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prompt_score = torch.tensor(prompt_score).to(device)

        def loss_fn(prediction, gold, attention, gradient, annotation, term_idx=None):
            # ordinal mse loss
            pred_score_fixed = prediction / prompt_score
            true_score_fixed = gold / prompt_score
            loss_first = F.mse_loss(pred_score_fixed, true_score_fixed)

            # attntion loss, attention mse
            selected_attn = attention
            fixed_annotation = torch.softmax((annotation - 1.0) * 1e+10, dim=2)
            loss_attn = F.mse_loss(selected_attn, fixed_annotation)

            # gradient loss
            fixed_annotation = torch.softmax((annotation - 1.0) * 1e+10, dim=2)
            fixed_annotation = (fixed_annotation.permute(0, 2, 1) * prompt_score).permute(0, 2, 1)
            loss_grad = F.mse_loss(gradient, fixed_annotation)

            loss = loss_first + _lambda_attn * loss_attn + _lambda_grad * loss_grad

            return loss

        return loss_fn
