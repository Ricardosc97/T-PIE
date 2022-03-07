# T-PIE

Pedestrian Intention Estimation using stacked Transformers Encoders. This model
is inspired by the [SF-GRU](https://github.com/aras62/SF-GRU) model and **use it
to generate the scene features**.

<p align="center">
  <img src="https://github.com/ricardosc97/T-PIE/blob/main/model.png?raw=true" title="hover text">
</p>

## Dependencies

This code is written and tested using:

- Python 3.6.9
- Numpy 1.19.5
- Pytorch 1.9.1
- Pytorch Lightning 1.4.9
- CUDA 10.2

And the code is trained and tested with [PIE](https://github.com/aras62/PIE)
dataset. The [SF-GRU](https://github.com/aras62/SF-GRU) is required to train the
model.

## Train and Test

As mentioned before, to run this model is required PIE and SF-GRU repos; The
train script is provided `train.py`. A sample for generating dataset is provided
below

```
from fv import FeatureVectors
from sf_gru import SFGRU
from pie_data import PIE
from t_pie import TPIE

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

data_opts = { 'seq_type': 'crossing',
              'data_split_type': 'random',
               ... }

model_opts = {'obs_input_type': ['local_box', 'local_context', 'pose', 'box', 'speed'],
              ...}

imdb = PIE(data_path=<path_to_pie>)
beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)

method_class = SFGRU()
train_val_data, data_types, data_sizes = method_class.get_data({'train': beh_seq_train}, model_opts=model_opts)
test_val_data, data_types, data_sizes = method_class.get_data({'test': beh_seq_test}, model_opts=model_opts)

train_dataset = FeatureVectors(train_val_data['train'], 'train')
test_dataset = FeatureVectors(test_val_data['test'], 'test', normalization=False)
```

Then prepare the model and use the `pl.Trainer()` Class.

```
model = TPIE(train_dataset, test_dataset)

trainer = pl.Trainer(
    gpus=-1,
    auto_lr_find = True,
    max_epochs=90
    )

trainer.fit(model)

trainer.test(model)
```

Using callbacks is optional

```
checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    filename="BestLoss-{epoch:02d}-{train_loss:.2f}",
    save_top_k=3,
    save_last = True,
    mode="min",
)

trainer = pl.Trainer(
    gpus=-1,
    auto_lr_find = True,
    max_epochs=90,
    callbacks=[checkpoint_callback]
    )
```

It is proposed to add the MCC metric to the results, being one of the most
balanced metric. The output after testing the model showed below

```
DATALOADER:0 TEST RESULTS
{'test_acc': 0.8327974081039429,
 'test_auc': 0.890505313873291,
 'test_f1': 0.6904761791229248,
 'test_mcc': 0.5808039903640747,
 'test_prec': 0.6373626589775085,
 'test_recall': 0.7532467246055603}
```

## Results

A Meidum article will be written showing the results in detail. For now, the
table below shows the results obtained for an 0.5s observation length and 2s
time to event.

<p align="center">
  <img src="https://github.com/ricardosc97/T-PIE/blob/main/results.png?raw=true" title="hover text">
</p>

## Other models

For academic purpose there are other models placed in the folder `/arqs`. Please
send email to ricardosc1997@gmail.com if you want to know more about them.

## Authors

- [Ricardo Silva](https://www.linkedin.com/in/ricardosc11/)

Please refers to [Amir Rasouli](https://aras62.github.io/) and
[Iuliia Kotseruba](https://ykotseruba.github.io/) authors of SF-GRU and PIE
