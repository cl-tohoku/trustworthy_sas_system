import pickle
from pathlib import Path
from glob import glob
import json
import sys
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm

sys.path.append("..")
from library.structure import Script
from preprocess.ys import YS
from preprocess.toppan import Toppan
from preprocess.fasttext import PreprocessFastText
from preprocess.bert import PreprocessBert
from preprocess.base import PreprocessBase
from library.util import Util
from collections import defaultdict

class PreprocessFinetuning:
    def __init__(self, prep_config, eval_config):
        self.config = prep_config
        self.eval_config = eval_config
        self.base = PreprocessBase(self.config)

    def load_updated_df_list(self, data_type):
        dir_expr = "{}_*_R".format(self.config.script_name)
        dir_expr = Path(self.eval_config.finetuning_dir) / data_type / dir_expr
        dir_list = glob(str(dir_expr))
        df_list_dict = defaultdict()
        for dir_path in dir_list:
            file_name = glob(str(Path(dir_path) / "*.pkl"))[0]
            df = pd.read_pickle(file_name, compression="xz")
            df_list_dict[df["Term"].unique()[0]] = df
        return df_list_dict


    def create_all_df(self, concat_df, max_id):
        all_df_list = []
        for term in concat_df["term"].unique():
            df = pd.DataFrame({"Sample_ID": list(range(max_id))})
            df["term"] = term
            df["heuristics"] = "All"
            all_df_list.append(df)
        return pd.concat(all_df_list).reset_index(drop=True)

    def concat_update_df_2(self, data_df, updated_df_dict, max_id):
        concat_list = []
        for term, updated_df in updated_df_dict.items():
            column_names = updated_df.columns[2:]
            part_df_list = []
            for name in column_names:
                sample_id_df = pd.DataFrame({"Sample_ID": updated_df[updated_df[name] == 1]["Sample_ID"].unique()})
                sample_id_df["heuristics"] = name
                part_df_list.append(sample_id_df)
            concat_df = pd.concat(part_df_list)
            concat_df["term"] = term
            concat_list.append(concat_df)
        update_df = pd.concat(concat_list).reset_index(drop=True)
        # all
        all_df = self.create_all_df(update_df, max_id)
        return pd.concat([update_df, all_df]).reset_index(drop=True)

    def load_integration_df(self, data_type):
        # load evaluated data df
        eval_df = Util.load_eval_df(self.eval_config, data_type, "attributions")

        # score_vector, input_ids, tokenized, annotation_char, token_types_ids, attention_mask
        data_df = eval_df[["Sample_ID", "Score_Vector", "Input_IDs", "Token", "Annotation_All", "Token_Type_IDs", "Attention_Mask"]]
        data_df.columns = ["Sample_ID", "score_vector", "input_ids", "token", "annotation_matrix", "token_type_ids", "attention_mask"]
        data_df = pd.concat([pd.DataFrame(data_df[data_df["Sample_ID"] == idx].iloc[0]).T for idx in data_df["Sample_ID"].unique()]).reset_index(drop=True)

        updated_df_dict = self.load_updated_df_list(data_type)
        heuristic_df = self.concat_update_df_2(data_df, updated_df_dict, eval_df["Sample_ID"].max())
        merged_df = pd.merge(data_df, heuristic_df, on="Sample_ID")
        return merged_df

    def to_pickle(self, class_dataset, prep_type, data_type, validation=None):
        file_name = "{}.{}.{}.finetuning.pkl".format(self.config.script_name, prep_type, data_type)

        # dump
        os.makedirs(self.config.dataset_dir, exist_ok=True)
        class_dataset.to_pickle(Path(self.config.dataset_dir) / file_name)

    def split_and_dump_pickle(self, script_df, prep_type, seed=187):
        # split train / dev
        train_dataset, valid_dataset = train_test_split(script_df, test_size=self.config.valid_size,
                                                       shuffle=True, random_state=seed)

        # dump
        self.to_pickle(train_dataset, prep_type, "train")
        self.to_pickle(valid_dataset, prep_type, "valid")

    def __call__(self):
        # load dataset & parse
        prompt = Util.load_prompt_config(self.config.prompt_path)
        train_df = self.load_integration_df("train")

        # split & dump
        self.base.dump_prompt(prompt)
        self.to_pickle(train_df, "bert", "train")
