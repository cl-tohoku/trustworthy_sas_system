import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
from captum.attr import *
from captum._utils.models.linear_model import SkLearnLasso, SkLearnLinearRegression, SkLearnLinearModel

sys.path.append("..")
from library.util import Util


class FeatureAttribution:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def embedding_func(model, token, parallel):
        # 埋め込み抽出
        if parallel:
            return model.module.bert.embeddings(token)
        else:
            return model.bert.embeddings(token)

    @staticmethod
    def reset_gradient(model):
        for param in model.parameters():
            param.grad = None

    @staticmethod
    def calc_vanilla_grad(model, token, args, target_idx=None, multiply=True, return_score=False, parallel=False,
                          step_size=None, training=False):
        # 予測
        input_emb = FeatureAttribution.embedding_func(model, token, parallel)
        input_emb.retain_grad()
        score = model(input_ids=input_emb, token_type_ids=args[0], attention_mask=args[1], inputs_embeds=True)

        # バッチ方向に足す(計算効率向上のため)
        batch_score = torch.sum(score, dim=0)
        idx_list = range(batch_score.shape[0]) if target_idx is None else [target_idx]

        # 項目ごとに入力に対する勾配を求める
        grad_list = []
        for idx in idx_list:
            grad, = torch.autograd.grad(batch_score[idx], input_emb, retain_graph=True, create_graph=True)
            grad_list.append(grad)
            # 勾配情報はリセット
            FeatureAttribution.reset_gradient(model)

        grad_tensor = torch.stack(grad_list).reshape(len(idx_list), *input_emb.shape)
        grad_tensor = grad_tensor.permute(1, 0, 2, 3)

        if return_score:
            return grad_tensor, score
        else:
            return grad_tensor

    @staticmethod
    def calc_int_grad_for_training(model, token, args, target_idx=None, step_size=128, parallel=False, internal_size=64):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 下準備（積分の区間の定義）、ガウス・ルシャンドルで数値積分する
        step_size_list = list(0.5 * np.polynomial.legendre.leggauss(step_size)[1])
        alpha_list = list(0.5 * (1 + np.polynomial.legendre.leggauss(step_size)[0]))

        # forward をやっておく
        input_emb = FeatureAttribution.embedding_func(model, token, parallel)
        score = model(input_emb, args[0], args[1], inputs_embeds=True)

        # 全ての項目か一つの項目か
        idx_list = range(score.shape[1]) if target_idx is None else [target_idx]

        grad_list = []
        for term_idx in idx_list:
            batch_list = []
            # バッチ入力を考慮する
            for batch_idx in range(input_emb.shape[0]):
                # 一つの項目、一つのデータに対して内部バッチで効率化する
                for slice_idx in range(0, step_size, internal_size):
                    # 毎回実行する必要がある
                    input_emb = FeatureAttribution.embedding_func(model, token, parallel)
                    data_emb = input_emb[batch_idx]
                    score = model(input_emb, args[0], args[1], inputs_embeds=True)
                    # ベースラインを定義
                    baseline_emb = torch.zeros(data_emb.shape, device=device)
                    # forward
                    sliced_alpha = alpha_list[slice_idx:slice_idx + internal_size]
                    alpha_emb_list = [baseline_emb + alpha * data_emb for alpha in sliced_alpha]
                    alpha_emb_tensor = torch.stack(alpha_emb_list)
                    alpha_type = args[0][batch_idx].unsqueeze(0).repeat(internal_size, 1)
                    alpha_mask = args[1][batch_idx].unsqueeze(0).repeat(internal_size, 1)
                    alpha_score = model(alpha_emb_tensor, alpha_type, alpha_mask, inputs_embeds=True)
                    alpha_score = torch.sum(alpha_score, dim=0)[term_idx]
                    # backward
                    alpha_grad, = torch.autograd.grad(alpha_score, alpha_emb_tensor, retain_graph=True, create_graph=True)
                    step_tensor = torch.tensor(step_size_list[slice_idx:slice_idx + internal_size], device=device).view(-1, 1, 1)
                    internal_grad = torch.sum(step_tensor * alpha_grad, dim=0) * data_emb
                    FeatureAttribution.reset_gradient(model)
                    yield internal_grad, score, batch_idx, term_idx

    @staticmethod
    def calc_int_grad(model, token, args, target_idx=None, multiply=True, step_size=128,
                    parallel=False, internal_size=64):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 下準備（積分の区間の定義）、ガウス・ルシャンドルで数値積分する
        step_size_list = list(0.5 * np.polynomial.legendre.leggauss(step_size)[1])
        alpha_list = list(0.5 * (1 + np.polynomial.legendre.leggauss(step_size)[0]))

        # forward をやっておく
        input_emb = FeatureAttribution.embedding_func(model, token, parallel)
        input_emb.retain_grad()
        score = model(input_emb, args[0], args[1], inputs_embeds=True)

        # 全ての項目か一つの項目か
        idx_list = range(score.shape[1]) if target_idx is None else [target_idx]

        grad_list = []
        for term_idx in idx_list:
            batch_list = []
            # バッチ入力を考慮する
            for batch_idx in range(input_emb.shape[0]):
                # ベースラインを定義
                data_emb = input_emb[batch_idx]
                baseline_emb = torch.zeros(data_emb.shape, device=device)
                grad = torch.zeros(data_emb.shape, device=device)
                # 一つの項目、一つのデータに対して内部バッチで効率化する
                for slice_idx in range(0, step_size, internal_size):
                    sliced_alpha = alpha_list[slice_idx:slice_idx + internal_size]
                    alpha_emb_list = [baseline_emb + alpha * data_emb for alpha in sliced_alpha]
                    alpha_emb_tensor = torch.stack(alpha_emb_list)
                    alpha_type = args[0][batch_idx].unsqueeze(0).repeat(internal_size, 1)
                    alpha_mask = args[1][batch_idx].unsqueeze(0).repeat(internal_size, 1)
                    alpha_score = model(alpha_emb_tensor, alpha_type, alpha_mask, inputs_embeds=True)
                    alpha_score = torch.sum(alpha_score, dim=0)[term_idx]
                    alpha_grad, = torch.autograd.grad(alpha_score, alpha_emb_tensor, retain_graph=True, create_graph=False)
                    step_tensor = torch.tensor(step_size_list[slice_idx:slice_idx + internal_size], device=device).view(-1, 1, 1)
                    internal_grad = torch.sum(step_tensor * alpha_grad, dim=0)
                    debug_grad = torch.sum(internal_grad, dim=1)
                    grad += internal_grad
                    FeatureAttribution.reset_gradient(model)
                batch_list.append(grad)
            grad_list.append(torch.stack(batch_list))

        # 入力を乗算するか
        if multiply:
            grad_list = [grad * input_emb for grad in grad_list]

        # バッチを先頭にする
        integrated_tensor = torch.stack(grad_list)
        integrated_tensor = integrated_tensor.permute(1, 0, 2, 3)

        return integrated_tensor

    @staticmethod
    def compress(tensor: torch.Tensor, summation=True):
        vanilla = tensor.squeeze(0).squeeze(0).cpu()
        vanilla = torch.sum(vanilla, dim=1).tolist() if summation else vanilla.tolist()
        return vanilla

