import torch
import torch.optim as optim
from dotenv import load_dotenv
import wandb
from libs.config import load_config
from libs.data import DataFetcher, get_train_dataloader, get_valid_dataloader
from libs.model import NeuronTransformer
from trainer import Trainer

import tkinter
from tkinter import filedialog


def train():
    root = tkinter.Tk()
    root.withdraw()
    config_file_path = filedialog.askopenfilename(parent=root, title='Please select config file', filetypes=[("Config files", "*.json"), ("All files", "*.*")])
    config = load_config(config_file_path)

    load_dotenv()

    model = NeuronTransformer(config)
    optimizer = optim.Adam(params=model.parameters(), lr=config.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataFetcher(get_train_dataloader(config), device=device)
    eval_dataloader = get_valid_dataloader(config) if config.validate else None

    writer = wandb.init(project=config.project_name, entity="toparkshin", name=config.run_name)

    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        writer=writer,
    )

    trainer.train()


if __name__ == "__main__":
    train()
