import pickle
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
#from pandas.core.common import SettingWithCopyWarning


sys.path.append("..")
from preprocess.base import PreprocessBase
from library.util import Util


class PreprocessMasking:
    def __init__(self, prep_config):
        self.config = prep_config
        self.mask_id = 101 # unk
        #warnings.simplefilter("ignore", SettingWithCopyWarning)

    def masking(self, df, masking_conf):
        masked_series = []
        df["masked_from"] = -1
        df_list = [df]

        for masking_span in masking_conf["span"]:
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
                    masked_ids_array[masked_bool_list] = self.mask_id
                    series = df.iloc[c_idx].copy()
                    series["input_ids"] = masked_ids_array.tolist()
                    series["masked_from"] = c_idx
                    masked_series.append(series)

            df_list.append(pd.DataFrame(masked_series))
            return pd.concat(df_list).reset_index(drop=True)

    def execute(self, mode):
        size = self.config.limitation
        train_df = Util.load_dataset_static(self.config.preprocess_name, size, "bert", "train", self.config.dataset_dir)
        valid_df = Util.load_dataset_static(self.config.preprocess_name, size, "bert", "valid", self.config.dataset_dir)
        test_df = Util.load_dataset_static(self.config.preprocess_name, size, "bert", "test", self.config.dataset_dir)

        # masking
        masking_conf = Util.load_masking_config(self.config.masking_path)
        masked_train_df = self.masking(train_df, masking_conf)
        masked_valid_df = self.masking(valid_df, masking_conf)

        masking_mode = "masking-{}".format(mode)
        self.to_pickle(masked_train_df, "train", masking_mode)
        self.to_pickle(masked_valid_df, "valid", masking_mode)
        self.to_pickle(test_df, "test", masking_mode)

        # load dataset & parse
        prompt = Util.load_prompt_config(self.config.prompt_path)
        self.dump_prompt(prompt, masking_mode)

    def to_pickle(self, df, data_type, mode):
        file_name = "{}.{}.bert.{}.{}.pkl".format(self.config.preprocess_name, self.config.limitation,
                                                  data_type, mode)

        # dump
        os.makedirs(Path(self.config.dataset_dir), exist_ok=True)
        df.to_pickle(Path(self.config.dataset_dir) / file_name)

    def dump_prompt(self, prompt, mode):
        os.makedirs(self.config.dataset_dir, exist_ok=True)
        file_name = "{}.{}.{}.prompt.yml".format(self.config.preprocess_name, self.config.limitation, mode)
        file_path = Path(self.config.dataset_dir) / file_name
        prompt.save(file_path)

    def __call__(self):
        self.execute("standard")
        self.execute("contamination")
