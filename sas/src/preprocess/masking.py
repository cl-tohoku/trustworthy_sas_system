import pickle
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from pandas.core.common import SettingWithCopyWarning


sys.path.append("..")
from preprocess.base import PreprocessBase
from library.util import Util

class PreprocessMasking:
    def __init__(self, prep_config):
        self.config = prep_config
        self.base = PreprocessBase(self.config)
        warnings.simplefilter("ignore", SettingWithCopyWarning)

    def to_pickle(self, df, data_type, masking_span):
        file_name = "{}.{}.{}.{}.pkl".format(self.config.script_name, "bert", data_type, masking_span)

        # dump
        os.makedirs(Path(self.config.dataset_dir) / "masking", exist_ok=True)
        df.to_pickle(Path(self.config.dataset_dir) / "masking" / file_name)

    def masking(self, df, masking_span):
        masked_series = []
        df["masked_from"] = -1

        for c_idx, (token_list, ids_list) in enumerate(zip(df["tokenized"], df["input_ids"])):
            # detect masking span
            token_list = ["[CLS]"] + token_list + ["[SEP]"]
            masked_bool_list = [False for _ in range(len(token_list))]
            for idx in range(len(token_list) - len(masking_span) + 1):
                checking_list = [token_list[idx + jdx] == masking_char for jdx, masking_char in enumerate(masking_span)]
                if all(checking_list):
                    for jdx in range(len(checking_list)):
                        masked_bool_list[idx + jdx] |= True

            # make masked data
            if any(masked_bool_list):
                masked_ids_array = np.array(ids_list)
                masked_ids_array[masked_bool_list] = 0
                series = df.iloc[c_idx].copy()
                series["input_ids"] = masked_ids_array.tolist()
                series["masked_from"] = c_idx
                masked_series.append(series)

        masked_df = pd.DataFrame(masked_series)
        return pd.concat([df, masked_df]).reset_index(drop=True)

    def execute(self, masking_span, previous_script_name):
        train_df = Util.load_dataset_static(previous_script_name, "bert", "train", self.config.dataset_dir)
        valid_df = Util.load_dataset_static(previous_script_name, "bert", "valid", self.config.dataset_dir)
        test_df = Util.load_dataset_static(previous_script_name, "bert", "test", self.config.dataset_dir)

        # masking
        masked_train_df = self.masking(train_df, masking_span)
        masked_valid_df = self.masking(valid_df, masking_span)

        print("train size: {}".format(len(masked_train_df)))
        print("valid size: {}".format(len(masked_valid_df)))

        self.to_pickle(masked_train_df, "train", masking_span)
        self.to_pickle(masked_valid_df, "valid", masking_span)
        self.to_pickle(test_df, "test", masking_span)

    def __call__(self, masking_span, previous_script_name):
        # load dataset & parse
        prompt = Util.load_prompt_config(self.config.prompt_path)
        self.base.dump_prompt(prompt)
        self.execute(masking_span, previous_script_name)