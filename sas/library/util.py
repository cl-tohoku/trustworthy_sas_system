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


class Util:
    def __init__(self):
        pass

    @staticmethod
    def load_preprocess_config(config_path):
        return PreprocessConfig.load(Path(config_path))

    @staticmethod
    def load_train_config(config_path):
        return TrainConfig.load(Path(config_path))

    @staticmethod
    def load_model_config(config_path):
        return ModelConfig.load(Path(config_path))

    @staticmethod
    def load_eval_config(config_path):
        return EvalConfig.load(Path(config_path))

    @staticmethod
    def load_prompt_config(config_path):
        return PromptConfig.load(Path(config_path))

    @staticmethod
    def load_sweep_config(sweep_path):
        """
        :param sweep_path: str, e.g. "config/sweep.yml"
        :return: dict
        """
        with open(sweep_path, "r") as f:
            return yaml.load(f)

    @staticmethod
    def load_dataset(config, data_type, finetuning=False):
        """
        :param config: Config
        :param data_type: str, "train" or "valid" or "test"
        :return: list(ScriptBert) or list(ScriptGlove)
        """
        prep_type = config.preprocessing_type
        if finetuning:
            file_name = "{}.{}.{}.finetuning.pkl".format(config.script_name, prep_type, data_type)
        elif config.validation:
            file_name = "{}.{}.{}.v{}.pkl".format(config.script_name, prep_type, data_type, config.validation_idx)
        else:
            file_name = "{}.{}.{}.pkl".format(config.script_name, prep_type, data_type)
        file_path = Path(config.dataset_dir) / file_name

        return pd.read_pickle(file_path)

    @staticmethod
    def load_vectors(config):
        """
        :param config: Config
        :return: vectors:
        """
        file_path = Path(config.dataset_dir) / "{}.fasttext.vectors.pkl".format(config.script_name)
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_prompt(config):
        file_name = "{}.prompt.yml".format(config.script_name)
        file_path = Path(config.dataset_dir) / file_name
        return Util.load_prompt_config(file_path)

    @staticmethod
    def select_model(model_config: ModelConfig, config):
        if config.preprocessing_type == "bert":
            return SimpleBert(model_config)
        elif config.preprocessing_type == "fasttext":
            vectors = Util.load_vectors(config)
            if config.model_name == "lstm":
                return LSTM(model_config, vectors)
            else:
                return FFNN(model_config, vectors)
        else:
            RuntimeError("Invalid preprocessing type")

    @staticmethod
    def replace_parallel_state_dict(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                k = k.replace('module.', '')
            new_state_dict[k] = v
        return new_state_dict

    @staticmethod
    def padding(array, max_length, padding_id):
        if len(array) > max_length:
            return array[:max_length]
        else:
            padding_length = max_length - len(array)
            return array + [padding_id for i in range(padding_length)]

    @staticmethod
    def save_model(model, config, finetuning=False):
        experiment_id = config.wandb_name if config.unique_id is None else config.unique_id
        if finetuning:
            file_name = "{}_{}.finetuning.state".format(config.script_name, experiment_id)
        elif config.validation:
            file_name = "{}_{}_v{}.state".format(config.script_name, experiment_id, config.validation_idx)
        else:
            file_name = "{}_{}.state".format(config.script_name, experiment_id)

        os.makedirs(config.model_dir, exist_ok=True)
        output_path = Path(config.model_dir) / file_name
        state_dict = model.state_dict()
        if config.parallel:
            state_dict = Util.replace_parallel_state_dict(state_dict)
        torch.save(state_dict, output_path)

    @staticmethod
    def load_model(config, model_config, finetuning=False):
        if finetuning:
            file_name = "{}_{}.finetuning.state".format(config.script_name, config.unique_id)
        elif config.validation:
            file_name = "{}_{}_v{}.state".format(config.script_name, config.unique_id, config.validation_idx)
        else:
            file_name = "{}_{}.state".format(config.script_name, config.unique_id)

        model = Util.select_model(model_config, config)
        state_dict = torch.load(Path(config.model_dir) / file_name)
        model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            model.cuda()

        return model

    @staticmethod
    def save_eval_df(dataframe, config, data_type, suffix, csv=True, finetuning=False):
        if finetuning:
            csv_file_name = "{}_{}_{}.finetuning.csv".format(config.unique_id, data_type, suffix)
            pkl_file_name = "{}_{}_{}.finetuning.pkl".format(config.unique_id, data_type, suffix)
        elif config.validation:
            v_idx = config.validation_idx
            csv_file_name = "{}_{}_{}.v{}.csv".format(config.unique_id, data_type, suffix, v_idx)
            pkl_file_name = "{}_{}_{}.v{}.pkl".format(config.unique_id, data_type, suffix, v_idx)
        else:
            csv_file_name = "{}_{}_{}.csv".format(config.unique_id, data_type, suffix)
            pkl_file_name = "{}_{}_{}.pkl".format(config.unique_id, data_type, suffix)

        os.makedirs(str(Path(config.eval_dir) / config.script_name), exist_ok=True)
        if csv:
            dataframe.to_csv((Path(config.eval_dir) / config.script_name / csv_file_name), index=False)
        dataframe.to_pickle((Path(config.eval_dir) / config.script_name / pkl_file_name))

    @staticmethod
    def load_eval_df(config, data_type, suffix):
        file_name = "{}_{}_{}.pkl".format(config.unique_id, data_type, suffix)
        df = pd.read_pickle((Path(config.eval_dir) / config.script_name / file_name))
        return df

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
    def attention_loss(prompt_score, anot_lambda=100.0):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prompt_score = torch.tensor(prompt_score).to(device)

        def loss_fn(input, target):
            pred_score, attention = input[0], input[1]
            true_score, annotation = target[0], target[1]
            pred_score_fixed = pred_score / prompt_score
            true_score_fixed = true_score / prompt_score
            annotation = torch.softmax((annotation - 1.0) * 10000, dim=2)
            attention_loss = F.mse_loss(annotation, attention)
            loss = F.mse_loss(pred_score_fixed, true_score_fixed) + anot_lambda * attention_loss
            return loss

        return loss_fn

    @staticmethod
    def to_tt(item, dtype=torch.int64, pad=False, value=0):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if pad:
            item = [torch.tensor(it, device=device) for it in item]
            item = torch.nn.utils.rnn.pad_sequence(item, padding_value=value, batch_first=True)
            item = item.to(dtype)
            return item
        else:
            return torch.tensor(item, device=device, dtype=dtype)

    @staticmethod
    def transform_annotation(annotation, attention_hidden_size):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tt_list = []
        max_length = max([len(term[0]) for term in annotation])
        for item in annotation:
            item_tt = torch.tensor(item,)
            pad_tt = torch.zeros((item_tt.shape[0], max_length - item_tt.shape[1]))
            tt_list.append(torch.cat([item_tt, pad_tt], dim=1))
        transformed = torch.stack(tt_list,).to(device)
        return transformed