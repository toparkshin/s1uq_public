import json
from dataclasses import dataclass


@dataclass
class Config:
    root_data_dir: str

    project_name: str
    run_name: str
    save_name: str

    data_prefix: str
    data_start_idx: int
    data_end_idx: int

    uq_mode: str
    n_logit_samples: int

    train_batch_size: int
    valid_batch_size: int
    test_batch_size: int
    lr: float

    n_class: int
    n_signal: int
    n_embd: int
    n_layer: int
    embd_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    num_iters: int
    num_workers: int
    seed: int

    validate: bool
    save_at_end: bool
    save_with_val_loss: bool

    eval_every: int
    visualize_every: int
    save_every: int


def _load_config(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return config_dict


def _validate_config(config):
    assert config.uq_mode in (
        "combined",
        "aleatoric",
        "epistemic",
    ), "ERROR: uq_mode must be one of 'combined', 'aleatoric' or 'epistemic'"


def load_config(config_path: str):
    config_dict = _load_config(config_path)
    config = Config(**config_dict)
    _validate_config(config)
    return config