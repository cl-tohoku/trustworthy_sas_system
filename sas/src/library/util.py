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
from captum.attr import IntegratedGradients

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
    def load_config(config_path, Config):
        return Config.load(Path(config_path))

    @staticmethod
    def load_dataset(config, data_type, script_name=None, mode=None):
        prep_type = config.preprocessing_type
        script_name = config.preprocess_name if script_name is None else script_name
        mode = config.mode if mode is None else mode
        file_name = "{}.{}.{}.{}.{}.pkl".format(script_name, config.limitation, prep_type, data_type, mode)
        file_path = Path(config.dataset_dir) / file_name

        return pd.read_pickle(file_path)

    @staticmethod
    def load_sf_dataset(sf_term, sf_idx, prompt_name, data_type, dataset_dir):
        file_name = "{}.{}-{}.{}.pkl".format(prompt_name, sf_term, data_type, "superficial")
        file_path = Path(dataset_dir) / file_name
        return pd.read_pickle(file_path)

    @staticmethod
    def load_dataset_static(prompt_name, data_type, mode, dataset_dir):
        file_name = "{}.{}.{}.pkl".format(prompt_name, data_type, mode)
        file_path = Path(dataset_dir) / file_name
        return pd.read_pickle(file_path)

    @staticmethod
    def load_vectors(config):
        file_path = Path(config.dataset_dir) / "{}.fasttext.vectors.pkl".format(config.script_name)
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_prompt(config):
        file_name = "{}.{}.prompt.yml".format(config.preprocess_name, config.mode)
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
    def load_model(config, model_config):
        script_name = config.script_name
        file_name = "{}.state".format(script_name)
        model = Util.select_model(model_config, config)
        state_dict = torch.load(Path(config.model_dir) / file_name)
        model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            model.cuda()
        return model

    @staticmethod
    def load_model_from_path(model_path, model_config):
        model = SimpleBert(model_config)
        state_dict = torch.load(Path(model_path))
        model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            model.cuda()
        return model

    @staticmethod
    def load_eval_df(config, data_type, suffix):
        file_name = "{}_{}.pkl".format(data_type, suffix)
        df = pd.read_pickle((Path(config.eval_dir) / config.script_name / file_name))
        return df

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
