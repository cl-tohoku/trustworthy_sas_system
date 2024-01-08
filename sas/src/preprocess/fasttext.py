import sys
import os

import torch
import wget
from pathlib import Path
from torchtext.data.utils import get_tokenizer
from dataclasses import asdict
from torchtext.vocab import Vectors
import pickle
from tqdm import tqdm

sys.path.append("..")
from library.structure import Script, ScriptFastText
from library.util import Util


class PreprocessFastText:
    def __init__(self, config):
        self.config = config
        self.tokenizer = lambda x: x.split()
        self.url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz"
        self.model_path = ".vector_cache/cc.ja.300.vec.gz"
        self.vec = None
        self.padding_id, self.unknown_id = None, None

    def download_vector_file(self):
        os.makedirs(".vector_cache", exist_ok=True)
        wget.download(self.url, out=self.model_path)

    def set_vectors(self):
        if self.config.download_ft:
            self.download_vector_file()
        self.vec = Vectors(name=self.model_path)
        # setting pad id
        self.padding_id = len(self.vec.itos)
        self.vec.vectors = torch.cat([self.vec.vectors, torch.zeros((1, self.vec.dim))], dim=0)
        self.vec.itos.append("<pad>")
        self.vec.stoi["<pad>"] = self.padding_id
        # setting unk id
        self.unknown_id = len(self.vec.itos)
        mean_vector = torch.mean(self.vec.vectors, dim=0).unsqueeze(0)
        self.vec.vectors = torch.cat([self.vec.vectors, mean_vector], dim=0)
        self.vec.itos.append("<unk>")
        self.vec.stoi["<unk>"] = self.unknown_id

    def encode(self, tokenized):
        ids = [self.vec.stoi[word] if word in self.vec.stoi.keys() else self.unknown_id for word in tokenized]
        return ids[:self.config.max_length]

    def to_ft_script(self, script):
        text = script.text
        tokenized = self.tokenizer(text)
        input_ids = self.encode(tokenized)

        ft_data = dict()
        ft_data["tokenized"] = tokenized
        ft_data["input_ids"] = input_ids
        ft_data.update(asdict(script))
        script_ft = ScriptFastText(**ft_data)
        script_ft.annotation_matrix = [word[:self.config.max_length] for word in script_ft.annotation_word]
        script_ft.unknown_id = self.unknown_id

        return script_ft

    def dump_vectors(self):
        os.makedirs(self.config.dataset_dir, exist_ok=True)
        file_name = "{}.fasttext.vectors.pkl".format(self.config.script_name)
        file_path = Path(self.config.dataset_dir) / file_name
        with open(str(file_path), "wb") as f:
            pickle.dump(self.vec, f)

    def __call__(self, scripts):
        self.set_vectors()
        ft_scripts = []
        for script in scripts:
            ft_scripts.append(self.to_ft_script(script))
        self.dump_vectors()
        return ft_scripts


