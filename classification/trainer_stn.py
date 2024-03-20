import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from data.mnist import MNIST
from standard_trainer import Trainer


class STNTrainer(Trainer):

    def _data_loader(self, train_data: Dataset, batch_size: int = 64):
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)

    @property
    def train_data_loader(self):
        dataset = MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        return self._data_loader(dataset)

    @property
    def test_data_loader(self):
        dataset = MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        return self._data_loader(dataset)

    def single_run(self, datas: Tensor, targets: Tensor, lr_scheduler=None, scaler=None) -> dict:
        datas, targets = datas.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(datas)

        # loss function
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()

        return {"losses": loss}

    def single_eval_run(self, datas: Tensor, targets: Tensor, lr_scheduler=None, scaler=None) -> dict:
        datas, targets = datas.to(self.device), targets.to(self.device)

        loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            pred = self.model(datas)
            losses = loss_fn(pred, targets).item()
        return {"test_losses": losses}


if __name__ == '__main__':
    from model.stn import STNet

    trainer = STNTrainer(
        model=STNet(),
        configs={"lr": 0.01}
    )
    trainer.train(epochs=10)
