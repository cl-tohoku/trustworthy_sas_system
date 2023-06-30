import sys
from torch.utils.data import Dataset, DataLoader
import torch

sys.path.append("..")
from library.util import Util


class DatasetForFastText(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        input_tuple = (self.dataset.iloc[item]["input_ids"],)
        score_tuple = (self.dataset.iloc[item]["score_vector"], self.dataset.iloc[item]["score"], self.dataset.iloc[item]["annotation_matrix"])
        return input_tuple, score_tuple


class DatasetForBert(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        input_tuple = (self.dataset.iloc[item]["input_ids"], self.dataset.iloc[item]["token_type_ids"], self.dataset.iloc[item]["attention_mask"])
        score_tuple = (self.dataset.iloc[item]["score_vector"], self.dataset.iloc[item]["score"], self.dataset.iloc[item]["annotation_matrix"])
        return input_tuple, score_tuple


class DatasetForFinetuning(Dataset):
    def __init__(self, dataset, heuristics):
        self.dataset = dataset
        self.heuristics = heuristics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        heuristics_data = self.dataset.iloc[item][self.heuristics]
        input_tuple = (self.dataset.iloc[item]["input_ids"], self.dataset.iloc[item]["token_type_ids"], self.dataset.iloc[item]["attention_mask"])
        score_tuple = (self.dataset.iloc[item]["score_vector"], 0, self.dataset.iloc[item]["annotation_matrix"], heuristics_data)
        return input_tuple, score_tuple


class Collate:
    @staticmethod
    def collate_ft(attention_hidden_size):
        ahs = attention_hidden_size

        def collate_fn(batch):
            it, st = list(zip(*batch))
            it, st = list(zip(*it)), list(zip(*st))
            it_t = (Util.to_tt(it[0],  pad=True), )
            st_t = (Util.to_tt(st[0], dtype=torch.float32), Util.to_tt(st[1], dtype=torch.float32),
                    Util.transform_annotation(st[2], ahs))
            return it_t, st_t

        return collate_fn

    @staticmethod
    def collate_bert(attention_hidden_size):
        ahs = attention_hidden_size

        def collate_fn(batch):
            it, st = list(zip(*batch))
            it, st = list(zip(*it)), list(zip(*st))
            it_t = (Util.to_tt(it[0],  pad=True), Util.to_tt(it[1], pad=True), Util.to_tt(it[2], pad=True))
            st_t = (Util.to_tt(st[0], dtype=torch.float32), Util.to_tt(st[1], dtype=torch.float32),
                    Util.transform_annotation(st[2], ahs))
            return it_t, st_t

        return collate_fn

    @staticmethod
    def collate_finetuning(attention_hidden_size):
        ahs = attention_hidden_size
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def collate_fn(batch):
            it, st = list(zip(*batch))
            it, st = list(zip(*it)), list(zip(*st))
            anot = Util.to_tt([torch.tensor(a, device=device).T for a in st[2]], pad=True, dtype=torch.float32).permute(0, 2, 1)
            it_t = (Util.to_tt(it[0],  pad=True), Util.to_tt(it[1], pad=True), Util.to_tt(it[2], pad=True))
            st_t = (Util.to_tt(st[0], dtype=torch.float32), Util.to_tt(st[1], dtype=torch.float32),
                    anot, Util.to_tt(st[3], dtype=torch.float32))
            return it_t, st_t

        return collate_fn


class Loader:
    @staticmethod
    def to_ft_dataloader(dataset, batch_size, attention_hidden_size):
        dataset_ft = DatasetForFastText(dataset)
        collate_fn = Collate.collate_ft(attention_hidden_size)
        return DataLoader(dataset_ft, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)

    @staticmethod
    def to_bert_dataloader(dataset, batch_size, attention_hidden_size):
        dataset_bert = DatasetForBert(dataset)
        collate_fn = Collate.collate_bert(attention_hidden_size)
        return DataLoader(dataset_bert, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

    @staticmethod
    def to_finetuning_dataloader(dataset, batch_size, attention_hidden_size, heuristic):
        dataset_finetuning = DatasetForFinetuning(dataset, heuristic)
        collate_fn = Collate.collate_finetuning(attention_hidden_size)
        return DataLoader(dataset_finetuning, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
