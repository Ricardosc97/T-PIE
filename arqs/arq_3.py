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
    def __init__(self, train_dataset, test_dataset, batch_size = 32, learning_rate = 5e-6, feature_d_model = 1065):
        super().__init__()
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset

        self.el_local_box = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first = True)
        self.te_local_box = nn.TransformerEncoder(self.el_local_box, num_layers=6)
        
        self.el_local_context = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first = True)
        self.te_local_context = nn.TransformerEncoder(self.el_local_context, num_layers=6)

        self.el_pose = nn.TransformerEncoderLayer(d_model=36, nhead=4, batch_first = True)
        self.te_pose = nn.TransformerEncoder(self.el_pose, num_layers=6)

        self.el_bbox = nn.TransformerEncoderLayer(d_model=4, nhead=4, batch_first = True)
        self.te_bbox = nn.TransformerEncoder(self.el_bbox, num_layers=6)

        self.el_speed = nn.TransformerEncoderLayer(d_model=1, nhead=1, batch_first = True)
        self.te_speed = nn.TransformerEncoder(self.el_speed, num_layers=6)

        self.f_d_model = feature_d_model

        self.fc = nn.Sequential(
            nn.Dropout(0.12), # if we are overfitting
            nn.Linear(1065, 512),
            nn.ReLU(),
            nn.Dropout(0.12), # if we are overfitting
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.12), # if we are overfitting
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.y_auc = torch.tensor(()).to(device)
        self.pred_auc = torch.tensor(()).to(device)

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

    
     # X shape: batch size, seq lenght, d model = 32, 14, 1065
    def forward(self, x):
        # Transformer

        # Separete inputs for de encoders
        local_box = x[:,:,0:512]        
        local_context = x[:,:,512:1024]
        pose = x[:,:,1024:1060]
        bbox = x[:,:,1060:1064]
        speed = x[:,:,1064:1065]

        # Transformers Outputs
        t_local_box = self.te_local_box(local_box.float())
        t_local_context = self.te_local_context(local_context.float())
        t_pose = self.te_pose(pose.float())
        t_bbox = self.te_bbox(bbox.float())
        t_speed = self.te_speed(speed.float())

        # Concatenate the transformers outputs along frames axis
        t_output_concatenated = torch.cat((
            t_local_box, 
            t_local_context, 
            t_pose, 
            t_bbox, 
            t_speed),             
            dim = 2).double().to(device)

        # Mean across frames. Note that dim changes with batch size dim
        mean = torch.mean(t_output_concatenated, dim = 1).type_as(x)
        # FC
        output = self.fc(mean.float()).type_as(x)
        # Output
        output = output.float().type_as(x)
        return output

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate) 
        return optimizer

    def training_step(self, batch, batch_idx):

        x, y  = batch

        # X shape (1, 14, 1065)
        # Forward
        y_hat = self(x)

        # Loss
        loss = F.binary_cross_entropy(y_hat, y)        

        # Metrics Calculation
        self.train_acc(y_hat, y.to(torch.uint8))
        self.train_f1(y_hat, y.to(torch.uint8))
        self.train_prec(y_hat, y.to(torch.uint8))
        self.train_recall(y_hat, y.to(torch.uint8))
        self.train_mcc(y_hat, y.to(torch.uint8))

        # Metrics Log
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_prec', self.train_prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mcc', self.train_mcc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False,on_epoch=True, prog_bar=True)
        self.log_dict({"step": self.current_epoch + 1}, on_step=False,on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx): 
        # The validation step is not longer used because the dataset implemented in the paper is with train and test set [0.6, 0.4]

        x, y  = batch
        # Forward
        y_hat = self(x)

        # Loss
        loss = F.binary_cross_entropy(y_hat, y)
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
        self.test_prec(predictions, y.to(torch.uint8))
        self.test_recall(predictions, y.to(torch.uint8))
        self.test_mcc(predictions, y.to(torch.uint8))
        self.conf_matrix(predictions, y.to(torch.uint8))

        # Metrics Log
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_prec', self.test_prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_mcc', self.test_mcc, on_step=False, on_epoch=True, prog_bar=True)


        self.y_auc = torch.cat((self.y_auc, torch.squeeze(y.to(torch.uint8), 1)), 0)
        self.pred_auc = torch.cat((self.pred_auc, torch.squeeze(predictions, 1)), 0)

        return

    def test_epoch_end(self, outputs):
        print('conf matrix', self.conf_matrix.compute())
        self.log('test_auc', 
            self.test_auc(
                self.pred_auc,
                self.y_auc.to(torch.uint8))
            , on_step=False, on_epoch=True, prog_bar=True)

        return

    def train_dataloader(self):
        train_loader = DataLoader(dataset = self.train_dataset, batch_size =  self.hparams.batch_size, shuffle = True, num_workers = 12)
        return train_loader

    def test_dataloader(self):
        test_loader = DataLoader(dataset = self.test_dataset, batch_size =  self.hparams.batch_size, shuffle = False, num_workers = 12)
        return test_loader
