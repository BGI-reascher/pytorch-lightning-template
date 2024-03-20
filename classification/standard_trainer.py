import math
import sys

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from utils import utils
from data.penn_fundan import PennFudanDataset, get_transform, collate_fn


class TrainerBase:
    def __init__(self, model: Module, configs: dict = {}, device=None):
        self.device = device or torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        self.configs = {
            "lr": 0.00001,
            "step_size": 3,
            "batch_size": 8
        }
        self.configs.update(configs)

    @property
    def optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=self.configs["lr"],
            momentum=0.9,
            weight_decay=0.0005
        )
        return optimizer

    @property
    def lr_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.configs["step_size"],
            gamma=0.1
        )

    @staticmethod
    def _data_loader(train_data: Dataset, batch_size: int = 3):
        data_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        return data_loader

    @property
    def train_data_loader(self):
        dataset = PennFudanDataset('./data/data/PennFudanPed', get_transform(train=True))
        return self._data_loader(dataset, self.configs["batch_size"])

    @property
    def test_data_loader(self):
        dataset = PennFudanDataset('./data/data/PennFudanPed', get_transform(train=False))
        return self._data_loader(dataset, self.configs["batch_size"])

    def single_run(self, datas: Tensor, targets: Tensor, lr_scheduler=None, scaler=None) -> dict:
        images = list(data.to(self.device) for data in datas)
        targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        self.optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            losses.backward()
            self.optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        return loss_dict


class Trainer(TrainerBase):

    def train(self, epochs: int = 10, print_freq: int = 10, scaler=None):
        """start train"""
        for epoch in range(epochs):
            self.model.train()

            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
            header = f"Epoch: [{epoch}]"

            lr_scheduler = None
            if epoch == 0:
                warmup_factor = 1.0 / 1000
                warmup_iters = min(1000, len(self.train_data_loader) - 1)

                lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
                )

            self.model.to(self.device)

            for datas, targets in metric_logger.log_every(self.train_data_loader, print_freq, header):
                loss_dict = self.single_run(datas, targets, lr_scheduler=lr_scheduler, scaler=scaler)

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    print(loss_dict_reduced)
                    sys.exit(1)

                metric_logger.update(**loss_dict_reduced)
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        return metric_logger


if __name__ == '__main__':
    from model.fast_rcnn import get_model_instance_segmentation

    trainer = Trainer(
        model=get_model_instance_segmentation(2)
    )
    trainer.train()
