import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
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
    def calc_int_grad(model, token, args, target_idx=None, multiply=True, step_size=8,
                      return_score=False, parallel=False, training=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 下準備
        step_size_list = list(0.5 * np.polynomial.legendre.leggauss(step_size)[1])
        alpha_list = list(0.5 * (1 + np.polynomial.legendre.leggauss(step_size)[0]))
        # forward
        input_emb = FeatureAttribution.embedding_func(model, token, parallel)
        input_emb.retain_grad()
        baseline_emb = torch.zeros(input_emb.shape, device=device)
        score = model(input_emb, args[0], args[1], inputs_embeds=True)
        idx_list = range(score.shape[1]) if target_idx is None else [target_idx]

        grad_list = [torch.zeros(input_emb.shape, device=device) for _ in range(score.shape[1])]
        for step_width, alpha in zip(step_size_list, alpha_list):
            alpha_emb = baseline_emb + alpha * input_emb
            alpha_score = model(alpha_emb, args[0], args[1], inputs_embeds=True)
            alpha_score = torch.sum(alpha_score, dim=0)
            for idx in idx_list:
                alpha_grad, = torch.autograd.grad(alpha_score[idx], alpha_emb, retain_graph=True, create_graph=training)
                alpha_grad = alpha_grad.contiguous()
                grad_list[idx] += alpha_grad * step_width
                # 勾配情報はリセット
                FeatureAttribution.reset_gradient(model)

        integrated_list = []
        for idx in idx_list:
            integrated = grad_list[idx]
            if multiply:
                integrated *= input_emb
            integrated_list.append(integrated)

        integrated_tensor = torch.stack(integrated_list)
        integrated_tensor = integrated_tensor.permute(1, 0, 2, 3)

        if return_score:
            return integrated_tensor, score
        else:
            return integrated_tensor

    def compress(self, tensor: torch.Tensor):
        vanilla = tensor.squeeze(0).squeeze(0).cpu()
        vanilla = torch.sum(vanilla, dim=1).tolist()
        return vanilla


"""
model.eval()
arg += (False, True)
input_emb = model.bert.embeddings(token)
baseline_emb = torch.zeros(input_emb.shape, device=self.device)

saliency = IntegratedGradients(model, multiply_by_inputs=True)
grad = saliency.attribute(input_emb, baselines=baseline_emb, target=target, additional_forward_args=arg,
                          n_steps=step_size, internal_batch_size=128)

attribution = self.to_vanilla(torch.sum(grad, dim=2))
return attribution
"""
