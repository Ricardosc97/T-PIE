
import numpy as np
import matplotlib.pyplot as plt

# PyTorch Lightning
import pytorch_lightning as pl

# PyTorch
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class AttentionModel(pl.LightningModule):
    def __init__(self, seq_len, embedding_size, batch_size = 32, lr = 1e-6 ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1065, nhead=15)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Sequential(
            nn.Linear(1065, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.batch_size = batch_size
        self.learning_rate = lr 

        # Metrics functions
        # Train Metrics
        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1()
        self.train_auc = torchmetrics.AUROC()
        self.train_prec = torchmetrics.Precision()
        self.train_recall = torchmetrics.Recall()
        self.train_mcc = torchmetrics.MatthewsCorrcoef(num_classes = 2)

        # Test Metrics
        self.test_acc = torchmetrics.Accuracy()
        self.test_f1 = torchmetrics.F1()
        self.test_auc = torchmetrics.AUROC()
        self.test_prec = torchmetrics.Precision()
        self.test_recall = torchmetrics.Recall()
        self.test_mcc = torchmetrics.MatthewsCorrcoef(num_classes = 2)

        self.conf_matrix = torchmetrics.ConfusionMatrix(num_classes=2)
        self.save_hyperparameters(ignore=[
            'train_dataset',
            'test_dataset'
        ])


    def forward(self, x):
        # Transformer
        t_output = self.transformer_encoder(x.permute(1,0,2).float())
        # Mean across frames 
        mean = torch.mean(t_output, dim = 0).type_as(x)
        # FC
        output = self.fc(mean.float()).type_as(x)
        # Output
        output = output.float().type_as(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) 
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y  = batch
        # Forward
        y_hat = self(x)

        # Loss
        loss = F.binary_cross_entropy(y_hat, y)

        self.log('train_loss', loss, on_step=False,on_epoch=True, prog_bar=True)
        
        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        # Forward
        predictions = self(x)
        
        # Loss
        loss = F.binary_cross_entropy(predictions, y)
        # Metrics
        self.test_acc(predictions, y.to(torch.uint8))
        self.test_f1(predictions, y.to(torch.uint8))
        self.test_auc(predictions, y.to(torch.uint8))
        self.test_prec(predictions, y.to(torch.uint8))
        self.test_recall(predictions, y.to(torch.uint8))
        self.test_mcc(predictions, y.to(torch.uint8))
        self.conf_matrix(predictions, y.to(torch.uint8))

        # Metrics Log
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_auc', self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_prec', self.test_prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_mcc', self.test_mcc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_conf_matrix', self.conf_matrix, on_step=False,on_epoch=False, prog_bar=False)
        self.log('test_loss', loss, on_step=False,on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs):
        print('conf matrix', self.conf_matrix.compute())

    def train_dataloader(self):
        train_loader = DataLoader(dataset = self.train_dataset, batch_size =  self.hparams.batch_size, shuffle = True, num_workers = 12)
        return train_loader

    def test_dataloader(self):
        test_loader = DataLoader(dataset = self.test_dataset, batch_size =  self.hparams.batch_size, shuffle = False, num_workers = 12)
        return test_loader
