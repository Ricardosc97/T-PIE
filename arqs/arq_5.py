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
    def __init__(self, batch_size = 32, lr = 1e-6, feature_d_model = 1065):
        super().__init__()
        self.el_local_box = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first = True)
        self.te_local_box = nn.TransformerEncoder(self.el_local_box, num_layers=3)
        
        self.el_local_context = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first = True)
        self.te_local_context = nn.TransformerEncoder(self.el_local_context, num_layers=3)

        self.el_pose = nn.TransformerEncoderLayer(d_model=1060, nhead=4, batch_first = True)
        self.te_pose = nn.TransformerEncoder(self.el_pose, num_layers=3)

        self.el_bbox = nn.TransformerEncoderLayer(d_model=1064, nhead=4, batch_first = True)
        self.te_bbox = nn.TransformerEncoder(self.el_bbox, num_layers=3)

        self.el_speed = nn.TransformerEncoderLayer(d_model=1065, nhead=15, batch_first = True)
        self.te_speed = nn.TransformerEncoder(self.el_speed, num_layers=3)

        # self.pe = PositionalEncoding(1066)

        self.f_d_model = feature_d_model


        self.fc = nn.Sequential(
            nn.Linear(feature_d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.y_auc = torch.tensor((),device=device)
        self.pred_auc = torch.tensor((), device=device)

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

        # Separete inputs in x_input and y_input per feature for transformer
        local_box = x[:,:,0:512]
        local_context = x[:,:, 512:1024]
        pose = x[:,:,1024:1060]
        bbox = x[:,:,1060:1064]
        speed = x[:,:,1064:1065]

        ### Lvl 1
        # Transformers local box
        output_te_lb = self.te_local_box(local_box.float())

        # Input for transformer local context
        input_te_local_context = torch.cat((local_context, output_te_lb), dim = 2).double().to(device)
        
        ### Lvl 2
        # Transformer local context
        output_te_lc = self.te_local_context(input_te_local_context.float())
        
        # Input for transformer pose
        input_te_pose  = torch.cat((pose, output_te_lc), dim = 2).double().to(device)
        
        ### Lvl 3
        # Transformer Pose
        output_te_pose = self.te_pose(input_te_pose.float())
        
        # Input for transformer bbox
        input_te_bbox = torch.cat((bbox, output_te_pose), dim = 2).double().to(device)

        ### Lvl 4 
        # Transformer Bbox
        output_te_bbox = self.te_bbox(input_te_bbox.float())

        # Input for transformer speed
        input_te_speed = torch.cat((speed, output_te_bbox), dim = 2).double().to(device)

        ### Lvl 5
        output_te_speed = self.te_speed(input_te_speed.float())

        # Mean across frames. Note that dim changes with batch size dim
        mean = torch.mean(output_te_speed, dim = 1).type_as(x)
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

    def test_dataloader(self, test_dataloader):
        return test_dataloader
