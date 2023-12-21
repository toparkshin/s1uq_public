import os
import os.path as osp
import pickle
from os.path import join as ospj
import torch
import tkinter
from tkinter import filedialog

from libs.config import load_config
from libs.data import get_test_dataloader
from libs.model import NeuronTransformer
from utils.infer_utils import infer

CHECKPOINTS_DIR = osp.abspath("checkpoints")


def get_model(config, device):
    model = NeuronTransformer(config)
    checkpoint_path = ospj(CHECKPOINTS_DIR, config.save_name)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.train()  # for MCD
    return model


if __name__ == "__main__":
    save_result = True
    num_mcd = 1000

    root = tkinter.Tk()
    root.withdraw()
    config_file_path = filedialog.askopenfilename(parent=root, title='Please select config file', filetypes=[("Config files", "*.json"), ("All files", "*.*")])

    config = load_config(config_file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config, device)
    model.train()
    dataloader = get_test_dataloader(config)

    aleatoric_result, epistemic_result = infer(model, dataloader, device, num_mcd, config)
    test_name = f"{config.run_name}_test"

    if save_result:
        out_dir = os.path.dirname(config_file_path)
        os.makedirs(out_dir, exist_ok=True)
        out_dict = {}
        out_dict["aleatoric_result"] = aleatoric_result
        out_dict["epistemic_result"] = epistemic_result
        out_dict["mouse_info"] = dataloader.dataset.mouse_info
        out_dict["trial_info"] = dataloader.dataset.trial_info
        out_dict["day_info"] = dataloader.dataset.day_info
        out_dict["stim_info"] = dataloader.dataset.stimulus
        out_dict["resp_info"] = dataloader.dataset.response
        out_dict["freq_info"] = dataloader.dataset.freq
        out_dict["label"] = dataloader.dataset.label
        with open(ospj(out_dir, test_name + "_Inference.pickle"), "wb") as f:
            pickle.dump(out_dict, f)

    print("DONE!")
