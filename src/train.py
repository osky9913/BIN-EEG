import torch
import torch.nn as nn
import tqdm, sys
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from customDataset import CustomDataset
from torch.utils.tensorboard import SummaryWriter
import logging

import pickle
from eegDataset import EegDataset

from evolutionNeuralNetworksUtils import  convert_chromosome_to_nn, create_chromosome, evaluate_nn, new_population, train_nn


if __name__ == "__main__":
    from datetime import datetime
    filename = "logs/"+datetime.now().strftime("%d-%m-%Y %H-%M-%S")#Setting the filename from current date and time

    file_handler = logging.FileHandler(filename=filename,mode='a')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig( handlers=handlers,
                    format="%(asctime)s, %(msecs)d %(name)s %(levelname)s [ %(filename)s-%(module)s-%(lineno)d ]  : %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.DEBUG)


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
    logging.debug("data_folder")
    logging.debug(str(data_folder))
    logging.debug("subjects_range")
    logging.debug(str(subjects_range))
    logging.debug("series_range")
    logging.debug(str(series_range))
    logging.debug("train_test_split_ratio")
    logging.debug(str(train_test_split_ratio))
    logging.debug("epochs")
    logging.debug(str(epochs))
    logging.debug("batch_size")
    logging.debug(str(batch_size))
    logging.debug("learning_rate")
    logging.debug(str(learning_rate))
    logging.debug("num_generations")
    logging.debug(str(num_generations))
    logging.debug("population_size")
    logging.debug(str(population_size))
    logging.debug("kernel_size")
    logging.debug(str(kernel_size))
    logging.debug("min_layers")
    logging.debug(str(min_layers))
    logging.debug("max_layers")
    logging.debug(str(max_layers))
    logging.debug("layer_sizes")
    logging.debug(str(layer_sizes))
    logging.debug("activations")
    logging.debug(str(activations))
    logging.debug("drop_out")
    logging.debug(str(drop_out))
    logging.debug("selection_rate")
    logging.debug(str(selection_rate))
    logging.debug("crossover_rate")
    logging.debug(str(crossover_rate))
    logging.debug("mutation_rate")
    logging.debug(str(mutation_rate))
    logging.debug("criterion")
    logging.debug(str(criterion))
    logging.debug("best_model_path")
    logging.debug(str(best_model_path))
    logging.debug("device")
    logging.debug(str(device))

    

    logging.debug("Loading dataset")
    train_data = EegDataset(data_folder, subjects_range, series_range, train_test_split_ratio, train=True)
    test_data = EegDataset(data_folder, subjects_range, series_range, train_test_split_ratio, train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,num_workers=4)
    logging.debug("Loaded dataset")

    count_individual = 0 

    input_size = train_data.input_data.shape[1]
    output_size = train_data.output_data.shape[1]
    logging.debug("input_size")
    logging.debug(input_size )
    logging.debug("output_size")
    logging.debug(output_size )
    population = [create_chromosome(layer_sizes=layer_sizes,kernel_sizes=kernel_size ,dropouts=drop_out,activations=activations,min_count_of_layer=min_layers,max_count_of_layer=max_layers,input_shape=(input_size,), output_shape=(output_size,)) for x in range(population_size) ]
    stats = []

    best_fitness = float('inf')
    best_chromosome = []
    for gen in range(num_generations):
        logging.debug("Gen ")
        logging.debug(gen)
        fitness_of_gen = []
        loss_of_gen = []
        for individual in population:
            logging.debug(count_individual+1)
            logging.debug("Architecture:")
            for layer in individual:
                logging.debug(layer)

            nn_model = convert_chromosome_to_nn(individual, input_shape=(input_size,), output_shape=(output_size,))
            nn_model = nn_model.to(device=device)
            optimizer = optim.Adam(nn_model.parameters(),learning_rate)
            loss = train_nn(nn_model=nn_model, train_loader=train_loader, epochs=epochs, device=device,optimizer=optimizer,criterion=criterion)
            fittness = evaluate_nn(nn_model=nn_model,test_loader=test_loader,device=device,criterion=criterion)
            if fittness < best_fitness:
                best_fitness = fittness
                logging.debug("New best fitness" )
                logging.debug(best_fitness )
                best_chromosome = individual
                logging.debug("New best chromosome")
                logging.debug(best_chromosome )
                best_model = nn_model
                torch.save(nn_model.state_dict(), best_model_path)
                inputs,labels = next(iter(train_loader))
                inputs, labels = inputs.to(device), labels.to(device)
                writer.add_graph(best_model,input_to_model=inputs)


            fitness_of_gen.append(fittness)
            loss_of_gen.append(loss)
            logging.debug(fittness)
            count_individual += 1

        thisdict = {}
        thisdict["Gen"] = gen
        thisdict["population"] = population
        thisdict["loss_of_gen"] = loss_of_gen
        thisdict["fitness_of_gen"] = fitness_of_gen

        stats.append(thisdict)
        logging.debug(stats)
        population = new_population(population=population,
                                    fitness=fitness_of_gen,
                                    num_parents=4,
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
