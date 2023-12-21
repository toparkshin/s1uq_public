import os.path as osp
import pickle
from functools import partial
from os.path import join as ospj
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .config import Config


def read_pickle(p: str):
    with open(p, "rb") as f:
        data = pickle.load(f)
    return data


class MouseBrainActivationDataset(Dataset):  # Total Dataset
    def __init__(
        self,
        config: Config,
        data_type: str = "train",
        read_data: bool = True,
        d: dict = None,
    ):
        self.config = config

        assert data_type in ("train", "valid", "test")

        if read_data:
            self.data_path = ospj(
                self.config.root_data_dir,
                f"{self.config.data_prefix}_{data_type}.pickle",
            )
            assert osp.exists(self.data_path)
            print(f"Reading {data_type} data...")
            d = read_pickle(self.data_path)
            print(f"Done reading {data_type} data!")

        assert d is not None

        self.data = d["data"]
        self.label = d["label"]

        self.stimulus = d["stimulus"]
        self.response = d["response"]
        self.freq = d["freq"]
        self.day_info = d["day_info"]
        self.trial_info = d["trial_info"]
        self.mouse_info = d["mouse_info"]

        assert len(self.data) == len(self.label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.preprocess_data(self.data[idx])
        label = self.label[idx]
        return data, label

    def get_all_items(self, idx):
        data = self.preprocess_data(self.data[idx])

        return {
            "data": data,
            "label": self.label[idx],
            "stimulus": self.stimulus[idx],
            "response": self.response[idx],
            "freq": self.freq[idx],
            "day_info": self.day_info[idx],
            "trial_info": self.trial_info[idx],
            "mouse_info": self.mouse_info[idx],
        }

    def preprocess_data(self, data):
        data = data[:, self.config.data_start_idx : self.config.data_end_idx]
        data = self.normalize(data)
        return data

    @staticmethod
    def normalize(data):
        max_vals = np.max(data, axis=1).reshape(-1, 1)
        min_vals = np.min(data, axis=1).reshape(-1, 1)
        dividend = max_vals - min_vals
        dividend = np.where(dividend == 0, np.Inf, dividend)
        data = (data - min_vals) / dividend
        return data


class MouseBrainActivationTestDataset(MouseBrainActivationDataset):
    def __init__(
        self,
        config: Config,
        data_type: str = "test",
        read_data: bool = True,
        d: dict = None,
    ):
        super().__init__(config, data_type, read_data, d)

    def __getitem__(self, idx):
        return self.get_all_items(idx)


def batch_pad_signal(batch, n_signal: int = 11):
    data_list, label_list, attention_masks = [], [], []

    max_n_neurons = -1
    data_idx = "data" if isinstance(batch[0], dict) else 0
    label_idx = "label" if isinstance(batch[0], dict) else 1
    for item in batch:
        max_n_neurons = max(max_n_neurons, item[data_idx].shape[0])

    for item in batch:
        data = item[data_idx]

        n_neurons = data.shape[0]
        n_padding_neurons = max_n_neurons - n_neurons

        padding = np.zeros((n_padding_neurons, n_signal), dtype=data.dtype)
        padded_data = np.concatenate([data, padding], axis=0)

        attention_mask = [True] * n_neurons + [False] * n_padding_neurons

        data_list.append(padded_data)
        label_list.append(item[label_idx])
        attention_masks.append(attention_mask)

    return data_list, label_list, attention_masks


def collate_fn(batch, n_signal: int = 11):
    data_list, label_list, attention_masks = batch_pad_signal(batch, n_signal)

    return (
        torch.tensor(data_list, dtype=torch.float32),
        torch.tensor(label_list, dtype=torch.float32),
        attention_masks,
    )


def test_collate_fn(batch, n_signal: int = 11):
    included_keys = [
        "stimulus",
        "response",
        "freq",
        "day_info",
        "trial_info",
        "mouse_info",
    ]

    ret_dict = {k: [] for k in ["data", "label"] + included_keys}

    data_list, label_list, _ = batch_pad_signal(batch, n_signal)
    ret_dict["data"] = torch.tensor(data_list, dtype=torch.float32)
    ret_dict["label"] = torch.tensor(label_list, dtype=torch.float32)

    for item in batch:
        for k, v in item.items():
            if k in included_keys:
                ret_dict[k].append(v)

    for k in ret_dict.keys():
        if k in included_keys:
            ret_dict[k] = np.asarray(ret_dict[k])

    return ret_dict


def get_train_dataloader(config: Config, d: dict = None):
    if d is None:
        data = MouseBrainActivationDataset(config, data_type="train")
    else:
        data = MouseBrainActivationDataset(
            config, data_type="train", read_data=False, d=d
        )

    data_loader_params = {
        "batch_size": config.train_batch_size,
        "shuffle": True,
        "num_workers": config.num_workers,
        "collate_fn": partial(collate_fn, n_signal=config.n_signal),
    }

    loader = DataLoader(data, **data_loader_params)
    return loader


def get_valid_dataloader(config: Config, d: dict = None):
    if d is None:
        data = MouseBrainActivationDataset(config, data_type="valid")
    else:
        data = MouseBrainActivationDataset(
            config, data_type="valid", read_data=False, d=d
        )

    data_loader_params = {
        "batch_size": config.valid_batch_size,
        "shuffle": False,
        "num_workers": 0,
        "collate_fn": partial(collate_fn, n_signal=config.n_signal),
    }

    loader = DataLoader(data, **data_loader_params)
    return loader


def get_test_dataloader(config: Config, d: dict = None):
    if d is None:
        data = MouseBrainActivationTestDataset(config, data_type="test")
    else:
        data = MouseBrainActivationTestDataset(config, data_type="test", read_data=False, d=d)

    data_loader_params = {
        "batch_size": config.test_batch_size,
        "shuffle": False,
        "num_workers": 0,
        "collate_fn": partial(test_collate_fn, n_signal=config.n_signal),
    }

    loader = DataLoader(data, **data_loader_params)
    return loader


class DataFetcher:
    def __init__(self, loader, device=None):
        self.loader = loader
        self.device = device

        self.iter = iter(self.loader)

    def _fetch_inputs(self):
        try:
            batch = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            batch = next(self.iter)
        return batch

    def __next__(self):
        batch = self._fetch_inputs()
        return batch
