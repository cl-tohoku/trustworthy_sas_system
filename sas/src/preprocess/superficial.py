import pickle
from pathlib import Path
import sys
import os
import numpy as np
import yaml
import pandas as pd
from tqdm import tqdm
import warnings
#from pandas.core.common import SettingWithCopyWarning
import MeCab
from collections import Counter
import itertools
from collections import defaultdict


sys.path.append("..")
from library.util import Util


class PreprocessSuperficial:
    def __init__(self, prep_config):
        self.config = prep_config

    def to_pickle(self, df, data_type):
        file_name = "{}.{}.superficial.pkl".format(self.config.preprocess_name, data_type)

        # dump
        os.makedirs(Path(self.config.dataset_dir), exist_ok=True)
        df.to_pickle(Path(self.config.dataset_dir) / file_name)

    def dump_prompt(self, prompt):
        os.makedirs(self.config.dataset_dir, exist_ok=True)
        file_name = "{}.superficial.prompt.yml".format(self.config.preprocess_name)
        file_path = Path(self.config.dataset_dir) / file_name
        prompt.save(file_path)


    def superficial_filter(self, sentence_series, score_series, superficial_cue=None):
        def filter_method(sentence, score):
            cond2 = superficial_cue in sentence
            cond3 = score > 0
            if not cond2 and not cond3:
                return True
            elif cond2 and cond3:
                return True
            else:
                return False

        filter_list = [filter_method(se, sc) for se, sc in zip(sentence_series, score_series)]
        return filter_list

    def superficial(self, df, superficial_cue=None):
        sentence_series = df["tokenized"].apply(lambda x: "".join(x))
        score_series = df["score_vector"].apply(lambda x: x[0])
        filter_sf = self.superficial_filter(sentence_series, score_series, superficial_cue)
        return df[filter_sf]

    def wakati(self, text):
        m = MeCab.Tagger()
        nodes = m.parse(text).split("\n")
        words = []
        filter_pos = ["名詞"]

        for node in nodes[:-2]:  # 最後の2行は ['EOS', ''] なので除外
            details = node.split("\t")
            pos_info = details[1].split(",")
            word = details[0]
            pos = pos_info[0]  # 品詞の主分類のみを取得
            if pos in filter_pos:
                words.append(word)

        return words

    def get_top_k_words(self, df, term_idx, k=5):
        # count word magnitude
        text_series = df["tokenized"].apply(lambda x: self.wakati("".join(x))).tolist()
        text_series = [list(set(text)) for text in text_series]
        text_list = list(itertools.chain.from_iterable(text_series))
        word_counter = Counter(text_list)

        # make rubric word list
        anno_series = df["annotation_matrix"].apply(lambda x: x[term_idx][1:-1]).tolist()
        token_series = df["tokenized"].apply(lambda x: x).tolist()
        anno_word_list = []
        for anno, token in zip(anno_series, token_series):
            anno_array, token_array = np.array(anno).astype(bool), np.array(token)
            anno_text = "".join(token_array[anno_array])
            if len(anno_text) > 0:
                anno_word_list.extend(self.wakati(anno_text))
        anno_word_list = list(set(anno_word_list))

        # remove word in rubric & sort
        _ = [word_counter.pop(word) for word in anno_word_list if word in word_counter]
        sorted_counter = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)

        chosen_word = [t[0] for t in sorted_counter if t[1] < (len(text_series) // 2)]
        return chosen_word[:k]

    def dump_manual(self, manual_dict):
        manual_dir = Path(self.config.dataset_dir) / "manual"
        os.makedirs(manual_dir, exist_ok=True)
        with open(manual_dir / "{}.yml".format(self.config.preprocess_name), "w", encoding='utf-8') as f:
            yaml.dump(dict(manual_dict), f, allow_unicode=True)

    def execute(self):
        train_df = Util.load_dataset_static(self.config.preprocess_name, "train", "standard", self.config.dataset_dir)
        valid_df = Util.load_dataset_static(self.config.preprocess_name, "valid", "standard", self.config.dataset_dir)
        test_df = Util.load_dataset_static(self.config.preprocess_name, "test", "standard", self.config.dataset_dir)
        prompt = Util.load_prompt_config(self.config.prompt_path)

        # get_top_k_words
        manual_dict = defaultdict(dict)
        for term_idx in range(prompt.scoring_item_num):
            top_k_words = self.get_top_k_words(train_df, term_idx)
            print(top_k_words)
            term = chr(65 + term_idx)
            for sf_idx, superficial_word in enumerate(top_k_words):
                superficial_train_df = self.superficial(train_df, superficial_word)
                superficial_valid_df = self.superficial(valid_df, superficial_word)
                self.to_pickle(superficial_train_df, "{}-{}-train".format(term, sf_idx))
                self.to_pickle(superficial_valid_df, "{}-{}-valid".format(term, sf_idx))
                self.to_pickle(test_df, "{}-{}-test".format(term, sf_idx))
                # set manual
                manual_dict[chr(term_idx + 65)][sf_idx] = superficial_word

        # output default dataset
        self.to_pickle(train_df, "train")
        self.to_pickle(valid_df, "valid")
        self.to_pickle(test_df, "test")

        # dump prompt
        self.dump_prompt(prompt)
        self.dump_manual(manual_dict)

    def __call__(self):
        # load dataset & parse
        self.execute()
