import os

import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax
from torchvision import models
import torch.nn as nn


def modified_res18():
    resnet = models.resnet18(pretrained=False, num_classes=10)
    resnet.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    resnet.maxpool = torch.nn.Identity()
    resnet.fc = torch.nn.Linear(512, 10, bias=False)
    return resnet


def conv_relu(c_in, c_out, kernel_size=(3, 3), padding=(1, 1)):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, bias=False),
        nn.ReLU()
    )

def conv_pool_act(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=(3, 3), padding=(1, 1), bias=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU()
    )

class MicroCnn(nn.Module):
    def __init__(self, width, c_out=10, scale_out=0.125):
        super().__init__()
        width = int(width)
        self.conv1 = nn.Conv2d(3, width, kernel_size=(3, 3), padding=(1, 1), bias=False) # (32, 32, 32)
        self.conv2 = conv_relu(width, 2*width, kernel_size=(3, 3)) # (64, 32, 32)
        self.conv3 = conv_pool_act(2*width, 4*width) # (128, 16, 16)
        self.conv4 = conv_pool_act(4*width, 8*width) # (256, 8, 8)
        self.conv5 = conv_pool_act(8*width, 16*width) # (512, 4, 4)
        self.conv6 = conv_relu(16*width, 16*width) # (512, 4, 4)
        self.pool7 = nn.MaxPool2d(kernel_size=4, stride=4) # (512, 1, 1)
        self.linear8 = nn.Linear(16*width, c_out, bias=False)
        self.scale_out = scale_out

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool7(x)
        x = x.reshape(x.size(0), x.size(1))
        x = self.linear8(x)
        x = self.scale_out * x
        return x


def get_model(model_name, is_imagenet=True):
    if is_imagenet:
       return  models.__dict__[model_name.lower()](pretrained=False, num_classes=1000)
       
    else: # cifar models
        if model_name.lower() == 'resnet18':
            return modified_res18()
        elif model_name.lower() == 'mini_cnn':
            return MicroCnn(width=32)


class LitModel(pl.LightningModule):
    def __init__(
        self,
        arch: str = "resnet18",
        learning_rate: float = 1e-1,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        schedule: str = 'step',
        dataset: str = 'cifar10',
    ):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = get_model(arch)

        self.train_acc = Accuracy()
        self.pred_accs = Accuracy()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.is_cifar5m = dataset.lower() == 'cifar5m'

        self.schedule = schedule

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        images, _ = batch
        return self.model(images)

    def process_batch(self, batch, stage="train"):
        images, labels = batch
        logits = self.forward(images)
        probs = softmax(logits, dim=1)
        loss = self.criterion(logits, labels)

        if stage == "train":
            self.train_acc(probs, labels)
        elif stage == "pred":
            self.pred_accs(probs, labels)
        else:
            raise ValueError("Invalid stage %s" % stage)

        return loss

    def training_step(self, batch, batch_idx: int):
        loss = self.process_batch(batch, "train")
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)

        # a bit hacky
        if self.is_cifar5m and batch_idx % 10 == 9: # every 10 batches
            sch = self.lr_schedulers()
            sch.step()
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss = self.process_batch(batch, "pred")
        self.log("pred_loss", loss)
        self.log(
            "pred_acc",
            self.pred_accs,
            on_step=False or self.is_cifar5m,
            on_epoch=True
        )

    def configure_optimizers(self):
        parameters = self.model.parameters()

        optimizer = torch.optim.SGD(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=0.9
        )

        if self.schedule == 'step':
            lr_scheduler = lr_sched.StepLR(optimizer, step_size=self.max_epochs//3 + 1, gamma=0.1)
        elif self.schedule == 'cos':
            lr_scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        else:
            raise NotImplementedError()

        return [optimizer], [lr_scheduler]
