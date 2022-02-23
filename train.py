from fv import FeatureVectors
from sf_gru import SFGRU
from pie_data import PIE
from t_pie import TPIE 


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dataset_opts = {'fstride': 1,
            'subset': 'default',
            'data_split_type': 'random',  # kfold, random, default
            'seq_type': 'crossing',
            'min_track_size': 75,
            'random_params': {
                'ratios': [0.6, 0.4],
                'val_data': False,
                'regen_data': False}
            }

imdb = PIE(data_path=<path_to_pie>)

model_opts = {'obs_input_type': ['local_box', 'local_context', 'pose', 'box', 'speed'],
              'enlarge_ratio': 1.5,
              'pred_target_type': ['crossing'],
              'obs_length': 15,  # Determines min track size
              'time_to_event': 60, # Determines min track size
              'dataset': 'pie',
              'normalize_boxes': True}

method_class = SFGRU()
beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)

train_val_data, data_types, data_sizes = method_class.get_data({'train': beh_seq_train}, model_opts=model_opts)
test_val_data, data_types, data_sizes = method_class.get_data({'test': beh_seq_test}, model_opts=model_opts)

train_dataset = FeatureVectors(train_val_data['train'], 'train')
test_dataset = FeatureVectors(test_val_data['test'], 'test', normalization=False)


model = TPIE(train_dataset, test_dataset)

#### Init ModelCheckpoint callback, monitoring 'train_loss'
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

trainer.fit(model)
result = trainer.test(model)