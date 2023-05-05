import torch.nn as nn
config = {
    'data_folder': '../dataset',
    'subjects_range': range(1, 2),
    'series_range': range(1, 2),
    'learning_rate': 0.0001,
    'epochs': 5,
    'num_generations' : 5,
    'num_individuals' : 8,
    'batch_size': 8192,
    'train_test_split_ratio': 0.7,
    'logs_folder': 'logs',
    'best_model_path': 'best_model.pth',
    'kernel_size': [1,3,5],
    'layer_sizes' : [8,16, 32, 64, 128, 256,512],
    'drop_out' : [0.0,0.0 ,0.1, 0.2],
    'activations' : [nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh],
    'criterion': nn.BCELoss(),
    'selection_rate' : 0.8,
    'mutation_rate': 0.3,
    'crossover_rate':0.3,
    'min_layers' : 2,
    'max_layers': 5
}
