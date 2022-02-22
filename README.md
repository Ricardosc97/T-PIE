# T-PIE
Thesis degree: Pedestrian Intention Estimation using stacked Transformers Encoders. This repo is inspired by the [SF-GRU](https://github.com/aras62/SF-GRU) model and **use it to generate the scene features**. 

![alt text](https://github.com/ricardosc97/T-PIE/model.png?raw=true)

## Dependencies 
This code is written and tested using: 

* Python 3.6.9
* Numpy 1.19.5
* Pytorch 1.9.1
* Pytorch Lightning 1.4.9
* CUDA 10.2

And the code is trained and tested with [PIE](https://github.com/aras62/PIE) dataset. The [SF-GRU](https://github.com/aras62/SF-GRU) is required to train the model. 

## Train 

As mentioned before, to run this model is required PIE and SF-GRU repos. 

@inproceedings{rasouli2017they,
  title={Pedestrian Action Anticipation using Contextual Feature Fusion in Stacked RNNs},
  author={Rasouli, Amir and Kotseruba, Iuliia and Tsotsos, John K},
  booktitle={BMVC},
  year={2019}
}

