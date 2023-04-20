from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix


class AttRnn(pl.LightningModule):
    def __init__(self, n_classes, lr = 0.001, l2 = 0.0):
        super().__init__()
        self.n_classes = n_classes
        self.lr = lr
        self.l2 = l2

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(5, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(10, 1, kernel_size=(5, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(1)
        )

        self.lstm = nn.LSTM(80, 64, 2, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(128, 128)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

        self.cm = ConfusionMatrix(task = 'multiclass', num_classes = n_classes)
        self.__last_cm__: Optional[torch.FloatTensor] = None

    def get_test_confusion_matrix(self) -> np.ndarray:
        return self.__last_cm__.detach().cpu().numpy()

    def on_fit_start(self):
        self = self.to(memory_format=torch.channels_last)

    def on_after_batch_transfer(self, batch, *args, **kwargs):
        batch[0] = batch[0].to(memory_format=torch.channels_last)
        return batch

    def forward(self, x):
        x = torch.transpose(x, 2, 3)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = torch.squeeze(x, 1)
        x, _  = self.lstm(x)
        xFirst = x[:, -1, :]
        query = self.linear(xFirst)
        x = F.scaled_dot_product_attention(query, x, x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self(x)
        val_loss = F.cross_entropy(pred, y)
        acc = torchmetrics.functional.accuracy(pred, y, 'multiclass', num_classes=self.n_classes)
        self.log("val_loss", val_loss)
        self.log('val_accuracy', acc, prog_bar = True)
        self.cm.update(pred, y)

    def on_validation_epoch_end(self):
        self.__last_cm__ = self.cm.compute()
        self.cm.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)
        return [optimizer], [scheduler]
