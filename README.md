# T-PIE
Thesis degree: Pedestrian Intention Estimation using stacked Transformers Encoders. This repo is inspired by the [SF-GRU](https://github.com/aras62/SF-GRU) model and **use it to generate the scene features**. 

<p align="center">
  <img src="https://github.com/ricardosc97/T-PIE/blob/main/model.png?raw=true" title="hover text">
</p>

## Dependencies 
This code is written and tested using: 

* Python 3.6.9
* Numpy 1.19.5
* Pytorch 1.9.1
* Pytorch Lightning 1.4.9
* CUDA 10.2

And the code is trained and tested with [PIE](https://github.com/aras62/PIE) dataset. The [SF-GRU](https://github.com/aras62/SF-GRU) is required to train the model. 

## Train and Test

As mentioned before, to run this model is required PIE and SF-GRU repos. 

```
from fv import FeatureVectors
from sf_gru import SFGRU
from pie_data import PIE
from t_pie import TPIE 


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

data_opts = { 'seq_type': 'crossing',
              'data_split_type': 'random',
               ... }

imdb = PIE(data_path=<path_to_pie>)

model_opts = {'obs_input_type': ['local_box', 'local_context', 'pose', 'box', 'speed'],
              ...}

method_class = SFGRU()
beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)

train_val_data, data_types, data_sizes = method_class.get_data({'train': beh_seq_train}, model_opts=model_opts)
test_val_data, data_types, data_sizes = method_class.get_data({'test': beh_seq_test}, model_opts=model_opts)

train_dataset = FeatureVectors(train_val_data['train'], 'train')
test_dataset = FeatureVectors(test_val_data['test'], 'test', normalization=False)


model = TPIE(train_dataset, test_dataset)
```

