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
    def __init__(self, config, model):
        self.model = model
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.config = config
        self.unknown_id = None

    def model_embeddings(self):
        if self.config.preprocessing_type == "bert":
            return self.model.bert.embeddings
        else:
            return self.model.embedding

    def forward_func_for_lime(self, token, token_type_ids=None, attention_mask=None):
        if token_type_ids is None:
            return self.model(token)
        else:
            return self.model(token, token_type_ids, attention_mask)

    # encode text indices into latent representations & calculate cosine similarity
    def exp_embedding_cosine_distance(self, original_inp, perturbed_inp, _, **kwargs):
        original_emb = torch.flatten(self.model_embeddings()(original_inp))
        perturbed_emb = torch.flatten(self.model_embeddings()(perturbed_inp))
        cosine_sim = F.cosine_similarity(original_emb, perturbed_emb, dim=0)
        distance = 1 - cosine_sim
        distance = torch.exp(-1 * (distance ** 2) / 0.75)
        return distance

    # binary vector where each word is selected independently and uniformly at random
    def bernoulli_perturb(self, token, **kwargs):
        probs = torch.ones_like(token) * 0.5
        return torch.bernoulli(probs).long()

    # remove absenst token based on the intepretable representation sample
    def interp_to_input(self, interp_sample, original_input, **kwargs):
        perturbed_input = original_input.detach().clone()
        perturbed_input[interp_sample.bool()] = self.unknown_id
        return perturbed_input

    def calc_lime(self, token, target, arg, unknown_id):
        self.unknown_id = unknown_id
        lasso_lime_base = LimeBase(
            self.forward_func_for_lime,
            interpretable_model=SkLearnLinearRegression(),
            similarity_func=self.exp_embedding_cosine_distance,
            perturb_func=self.bernoulli_perturb,
            perturb_interpretable_space=True,
            from_interp_rep_transform=self.interp_to_input,
            to_interp_rep_transform=None
        )

        attrs = lasso_lime_base.attribute(
            token,  # add batch dimension for Captum
            target=target,
            additional_forward_args=arg,
            n_samples=100,
            show_progress=False,
        ).squeeze(0) * -1.0
        return attrs.to("cpu").tolist()

    def predict_baseline(self, embedding):
        with torch.no_grad():
            return self.model(input_emb=embedding).squeeze(0).long().to("cpu").tolist()

    def calc_gradient(self, token, target, arg, multiply=True):
        arg += (False, True)
        input_emb = self.model_embeddings()(token)
        baseline_emb = torch.zeros(input_emb.shape, device=self.device)

        # Support for Bi-LSTM
        if self.config.model_name == "lstm":
            self.model.train()

        step_size = self.config.step_size
        saliency = IntegratedGradients(self.model, multiply_by_inputs=True)
        grad = saliency.attribute(input_emb, baselines=baseline_emb, target=target,
                                  additional_forward_args=arg, n_steps=step_size, internal_batch_size=128)
        attribution = torch.sum(grad, dim=2)

        # calc baseline output
        baseline_pred_tensor = self.model(baseline_emb, *arg)
        baseline_pred = baseline_pred_tensor.squeeze(0).cpu().tolist()[target]

        grad, input_emb, attribution = self.to_vanilla(grad), self.to_vanilla(input_emb), self.to_vanilla(attribution)

        self.model.eval()
        return attribution, grad, input_emb, baseline_pred


    def to_vanilla(self, tensor):
        vanilla = tensor.squeeze(0).cpu().tolist()
        return vanilla

