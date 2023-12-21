import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from libs.config import Config
from utils import (
    AverageMeter,
    ModelEval,
    ModelSaver,
    batch_visualize,
    get_num_parameters,
    save_checkpoint,
    set_seeds,
    calculate_acc,
)


class Trainer:
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        optimizer,
        device,
        train_dataloader,
        eval_dataloader=None,
        writer=None,
    ):
        self.config = config

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_dataloader = train_dataloader

        self.eval_dataloader = eval_dataloader
        self.writer = writer

        self.model_saver = ModelSaver()

        self.train_loss = AverageMeter("Train Loss")
        self.train_acc = AverageMeter("Train Accuracy")
        self.eval_loss = AverageMeter("Eval Loss")
        self.eval_acc = AverageMeter("Eval Accuracy")

        self.log = self.writer is not None
        self.eval = self.config.eval_every > 0
        self.viz = self.config.visualize_every > 0
        self.save = self.config.save_every > 0

        self.global_step = 0
        self.step = 0

    def on_train_begin(self):
        # Model Parallel
        self.model.to(self.device)

        # Other initializations
        set_seeds(self.config.seed)

        # Initial logging
        print(f"Number of Total Parameters: {get_num_parameters(self.model)}")

        self.writer.config.update(dataclasses.asdict(self.config))

    def on_train_end(self):
        if self.config.save_at_end:
            torch.save(self.model.state_dict(), f"checkpoints/{self.config.save_name}")

    def on_train_batch_begin(self, it, batch):
        # Visualize
        if self.viz and (it + 1) % self.config.visualize_every == 0:
            batch_visualize(batch)

    def on_train_batch_end(self, it, batch):
        # Update steps
        self.global_step += len(batch[0])
        self.step += 1

        if self.log:
            self.writer.log(
                {
                    "train/loss": self.train_loss.val,
                    "train/accuracy": self.train_acc.val,
                },
                step=self.global_step,
            )

        if self.eval and (it + 1) % self.config.eval_every == 0:
            with ModelEval(self.model) as m:
                self.evaluate(m)

        if self.save and (it + 1) % self.config.save_every == 0:
            save_checkpoint(self.model)

    def on_eval_begin(self):
        return

    def on_eval_end(self):
        if self.log:
            self.writer.log(
                {
                    "eval/loss": self.eval_loss.avg,
                    "eval/accuracy": self.eval_acc.avg,
                },
                step=self.global_step,
            )

        if self.config.save_with_val_loss:
            self.model_saver(
                self.eval_loss.avg,
                self.model,
                f"checkpoints/{self.config.run_name}_val_loss_{self.eval_loss.avg}.pth",
            )

        self.eval_loss.reset()
        self.eval_acc.reset()

    def on_eval_batch_begin(self):
        return

    def on_eval_batch_end(self):
        return

    def train(self):
        self.on_train_begin()

        pbar = tqdm(range(self.config.num_iters))

        for it in pbar:
            batch = next(self.train_dataloader)
            x, y, attention_mask = batch
            x, y = x.to(self.device), y.to(self.device)
            self.on_train_batch_begin(it, (x, y))

            out, _ = self.model(x, training=True)
            loss = F.cross_entropy(out, y.to(torch.long))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.train_loss.update(float(loss))
            self.train_acc.update(calculate_acc(out, y))

            pbar.set_description(
                f"Train: Iter - {it} Loss - {float(self.train_loss.val):.8f} Accuarcy - {float(self.train_acc.val)}",
            )

            self.on_train_batch_end(it, (x, y))

        self.on_train_end()

    def evaluate(self, model):
        if self.eval_dataloader is None:
            return

        self.on_eval_begin()

        pbar = enumerate(tqdm(self.eval_dataloader))

        for i, batch in pbar:
            self.on_eval_batch_begin()

            x, y, attention_mask = batch
            x, y = x.to(self.device), y.to(self.device)
            out, _ = model(x)
            loss = F.cross_entropy(out, y.to(torch.long))

            self.eval_loss.update(float(loss))
            self.eval_acc.update(calculate_acc(out, y))

            self.on_eval_batch_end()

        self.on_eval_end()
