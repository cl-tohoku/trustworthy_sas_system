from dataclasses import dataclass, field
import dataclasses
import pathlib
import yaml
import inspect
import numpy as np
import torch


class YAML:
    def save(self, config_path: pathlib.Path):
        """ Export config as YAML file """
        assert config_path.parent.exists(), f'directory {config_path.parent} does not exist'

        def convert_dict(data):
            for key, val in data.items():
                if isinstance(val, pathlib.Path):
                    data[key] = str(val)
                if isinstance(val, dict):
                    data[key] = convert_dict(val)
            return data

        with open(config_path, 'w') as f:
            yaml.dump(convert_dict(dataclasses.asdict(self)), f)

    @classmethod
    def load(cls, config_path: pathlib.Path):
        """ Load config from YAML file """
        assert config_path.exists(), f'YAML config {config_path} does not exist'

        def convert_from_dict(parent_cls, data):
            for key, val in data.items():
                child_class = parent_cls.__dataclass_fields__[key].type
                if child_class == pathlib.Path:
                    data[key] = pathlib.Path(val)
                if inspect.isclass(child_class) and issubclass(child_class, YAML):
                    data[key] = child_class(**convert_from_dict(child_class, val))
            return data

        with open(config_path) as f:
            config_data = yaml.full_load(f)
            # recursively convert config item to YamlConfig
            config_data = convert_from_dict(cls, config_data)
            return cls(**config_data)

    def update(self, new: dict):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)


# config dataclasses

@dataclass
class PreprocessConfig(YAML):
    preprocess_name: str = "Y14_1213"
    script_path: str = "data/data_ys/Y14/Y14_1-2_1_3.json"
    prompt_path: str = "config/prompt/test_prompt.yml"
    dataset_dir: str = "data/pickle/Y14"
    dataset_type: str = "ys"
    download_ft: bool = False
    max_length: int = 512
    test_size: float = 0.2
    valid_size: float = 0.2

@dataclass
class SVPreprocessConfig(YAML):
    preprocess_name: str = "Y14_1213"
    prompt_path: str = "config/prompt/test_prompt.yml"
    dataset_dir: str = "data/pickle/Y14"
    # superficial mode
    sf_term: str = None
    sf_idx: int = None
    target_score: int = 2
    cluster_dir: str = "data/cluster/Y14"
    script_name: str = "Y14_1213_XXXX"
    threshold: float = 0.1
    sampling_size: int = 10
    prev_mode: str = "superficial"
    prev_script_name: str = None


@dataclass
class PromptConfig(YAML):
    scoring_item_num: int = 4
    deduction_eos: bool = False
    deduction_spl: bool = False
    max_scores: list = field(default_factory=lambda: [2, 5, 3, 6])


@dataclass
class TrainConfig(YAML):
    # I/O
    preprocess_name: str = "Y14_1213"
    script_name: str = "Y14_1213_standard"
    dataset_dir: str = "data/pickle/Y14"
    model_dir: str = "data/model"
    model_config_path: str = ""
    preprocessing_type: str = "bert"
    model_name: str = "bert-test"
    unique_id: str = None
    # training parameters
    loss: str = "mse"
    learning_rate: float = 0.000005
    batch_size: int = 32
    epoch: int = 50
    parallel: bool = True
    mode: str = "standard"
    # wandb
    wandb: bool = True
    wandb_entity: str = "ekupura"
    wandb_project_name: str = "jsas"
    wandb_group: str = "bert"
    # sweep
    sweep: bool = False
    sweep_config_path: str = ""
    sweep_count: int = 50
    # superficial mode
    sf_term: str = None
    sf_idx: int = None


@dataclass
class ModelConfig(YAML):
    dropout_rate: float = 0.5
    hidden_size: int = 128
    attention_hidden_size: int = 512
    batch_size: int = 32
    output_size: int = 4


@dataclass
class EvalConfig(YAML):
    preprocess_name: str = "Y14_1213"
    script_name: str = "Y14_1-2_1_3"
    dataset_dir: str = "data/pickle/Y14"
    model_dir: str = "data/model"
    model_config_path: str = ""
    eval_dir: str = "data/eval"
    preprocessing_type: str = "bert"
    model_name: str = "bert-test"
    # attribution
    attribution: bool = False
    attr_config_path: str = "config/template/attribution.yml"
    # clustering
    n_clusters: list = field(default_factory=list)
    cluster_dir: str = "data/cluster/Y14"
    point_type: str = "attribution"
    gold_cluster_dir: str = ""
    step_size: int = 512
    # finetune
    mode: str = "standard"
    # superficial mode
    sf_term: str = None
    sf_idx: int = None


@dataclass
class AttrConfig(YAML):
    method: str = "lime"
    alpha: float = 0.00


# script dataclasses

@dataclass
class Script:
    Sample_ID: int = 0
    text: str = ""
    score: int = 0
    score_vector: list = field(default_factory=list)
    annotation_matrix: list = field(default_factory=list)
    annotation_word: list = field(default_factory=list)
    annotation_char: list = field(default_factory=list)
    unknown_id: int = 0


@dataclass
class ScriptBert(Script):
    tokenized: list = field(default_factory=list)
    input_ids: list = field(default_factory=list)
    token_type_ids: list = field(default_factory=list)
    attention_mask: list = field(default_factory=list)


@dataclass
class ScriptFastText(Script):
    tokenized: list = field(default_factory=list)
    input_ids: list = field(default_factory=list)