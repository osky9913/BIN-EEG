import torch
import torch.nn as nn
import tqdm, sys
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from customDataset import CustomDataset
from torch.utils.tensorboard import SummaryWriter

import pickle

from evolutionNeuralNetworksUtils import  convert_chromosome_to_nn, create_chromosome, evaluate_nn, new_population, train_nn


if __name__ == "__main__":
    config = config.config
    writer = SummaryWriter("torchlogs/")
    data_folder = config['data_folder']
    subjects_range = config['subjects_range']
    series_range = config['series_range']
    train_test_split_ratio = 0.7
    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_generations = config['num_generations']
    population_size = config['num_individuals']
    kernel_size = config['kernel_size']
    min_layers = config['min_layers']
    max_layers = config['max_layers']
    layer_sizes= config['layer_sizes']
    activations = config['activations']
    drop_out = config['drop_out']
    selection_rate = config['selection_rate']
    crossover_rate = config['crossover_rate']
    mutation_rate = config['mutation_rate']
    criterion = config['criterion']
    best_model_path = config['best_model_path']
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    #device = 'cpu'
    print("writer",writer)
    print("data_folder",data_folder)
    print("subjects_range",subjects_range)
    print("series_range",series_range)
    print("train_test_split_ratio",train_test_split_ratio)
    print("epochs",epochs)
    print("batch_size",batch_size)
    print("learning_rate",learning_rate)
    print("num_generations",num_generations)
    print("population_size",population_size)
    print("kernel_size",kernel_size)
    print("min_layers",min_layers)
    print("max_layers",max_layers)
    print("layer_sizes",layer_sizes)
    print("activations",activations)
    print("drop_out",drop_out)
    print("selection_rate",selection_rate)
    print("crossover_rate",crossover_rate)
    print("mutation_rate",mutation_rate)
    print("criterion",criterion)
    print("best_model_path",best_model_path)
    print("device",device)
    

    train_data = CustomDataset(data_folder, subjects_range, series_range, train_test_split_ratio, train=True)
    test_data = CustomDataset(data_folder, subjects_range, series_range, train_test_split_ratio, train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    count_individual = 0 

    input_size = train_data.input_data.shape[1]
    output_size = train_data.output_data.shape[1]
    print("input_size ", input_size )
    print("output_size ", output_size )
    population = [create_chromosome(layer_sizes=layer_sizes,kernel_sizes=kernel_size ,dropouts=drop_out,activations=activations,min_count_of_layer=min_layers,max_count_of_layer=max_layers,input_shape=(input_size,), output_shape=(output_size,)) for x in range(population_size) ]
    stats = []

    best_fitness = float('inf')
    best_chromosome = []
    for gen in range(num_generations):
        print("Gen ", gen)
        fitness_of_gen = []
        loss_of_gen = []
        for individual in population:
            print(count_individual+1)
            print("Architecture:",)
            for layer in individual:
                print(" ", layer)
            nn_model = convert_chromosome_to_nn(individual, input_shape=(input_size,), output_shape=(output_size,))
            nn_model = nn_model.to(device=device)
            optimizer = optim.Adam(nn_model.parameters(),learning_rate)
            loss = train_nn(nn_model=nn_model, train_loader=train_loader, epochs=epochs, device=device,optimizer=optimizer,criterion=criterion)
            fittness = evaluate_nn(nn_model=nn_model,test_loader=test_loader,device=device,criterion=criterion)
            if fittness < best_fitness:
                best_fitness = fittness
                print("New best fitness" ,best_fitness )
                best_chromosome = individual
                print("New best chromosome" ,best_chromosome )
                best_model = nn_model
                torch.save(nn_model.state_dict(), best_model_path)
                inputs,labels = next(iter(train_loader))
                inputs, labels = inputs.to(device), labels.to(device)

                writer.add_graph(best_model,input_to_model=inputs)

            fitness_of_gen.append(fittness)
            loss_of_gen.append(loss)
            print(fittness)
            count_individual += 1

        thisdict = {}
        thisdict["Gen"] = gen
        thisdict["population"] = population
        thisdict["loss_of_gen"] = loss_of_gen
        thisdict["fitness_of_gen"] = fitness_of_gen

        stats.append(thisdict)
        population = new_population(population=population,
                                    fitness=fitness_of_gen,
                                    num_parents=2,
                                    mutation_rate=mutation_rate,
                                    layer_sizes=layer_sizes,
                                    kernel_sizes=kernel_size,
                                    dropouts=drop_out,
                                    activations=activations,
                                    input_shape=(input_size,),
                                    output_shape=(output_size,)
                                    )
    
        with open('results.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

   # with open('filename.pickle', 'rb') as handle:
   #     stats_pickle = pickle.load(handle)
    print("STATS:", stats)
    #plot_fitness(stats=stats)
