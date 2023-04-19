import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from functools import reduce
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np

import tqdm
import config
from customDataset import CustomDataset
import random


config = config.config

def create_chromosome(layer_sizes, kernel_sizes, dropouts, activations, min_count_of_layer, max_count_of_layer, input_shape, output_shape):
    count_of_layer = random.randint(min_count_of_layer, max_count_of_layer)
    chromosome = []
    input_size, = input_shape
    output_size, = output_shape
    for i in range(count_of_layer):
        layer_size = random.choice(layer_sizes)
        valid_kernel_sizes = [k for k in kernel_sizes if k <= input_size]
        if not valid_kernel_sizes:
            break
        kernel_size = random.choice(valid_kernel_sizes)
        dropout = random.choice(dropouts)
        activation = random.choice(activations).__name__

        chromosome.append({
            'layer_size': layer_size,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'activation': activation
        })

        input_size = input_size - (kernel_size - 1)

    return chromosome

def get_valid_kernel_sizes(kernel_sizes, input_size):
    return [k for k in kernel_sizes if k <= input_size]

def mutate_chromosome(chromosome, layer_sizes, kernel_sizes, dropouts, activations, mutation_rate, input_size):
    mutated_chromosome = []
    
    for layer_info in chromosome:
        layer_size = layer_info['layer_size']
        kernel_size = layer_info['kernel_size']
        dropout = layer_info['dropout']
        activation = layer_info['activation']
        
        if random.random() < mutation_rate:
            layer_size = random.choice(layer_sizes)
            
        valid_kernel_sizes = get_valid_kernel_sizes(kernel_sizes, input_size)
        if valid_kernel_sizes and random.random() < mutation_rate:
            kernel_size = random.choice(valid_kernel_sizes)
            
        if random.random() < mutation_rate:
            dropout = random.choice(dropouts)
            
        if random.random() < mutation_rate:
            activation = random.choice(activations).__name__
            
        mutated_layer = {
            'layer_size': layer_size,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'activation': activation
        }
        
        mutated_chromosome.append(mutated_layer)
        input_size = input_size - (kernel_size - 1)
        
    return mutated_chromosome

def crossover(parent1, parent2, crossover_rate=0.5):
    child = []
    
    # Check if the chromosome lengths match
    if len(parent1) != len(parent2):

        #raise ValueError("Parent chromosomes must have the same length")
        if len(parent1)> len(parent2):
            return parent1
        else:
            return parent2
    
    # Iterate through each layer in the parent chromosomes and perform crossover
    for i in range(len(parent1)):
        # Select the layer from one of the parents randomly
        layer = parent1[i] if random.random() < crossover_rate else parent2[i]
        
        # Check if the previous and next layers have matching kernel sizes and layer sizes
        prev_layer_size = parent1[i-1]['layer_size'] if i > 0 else None
        prev_kernel_size = parent1[i-1]['kernel_size'] if i > 0 else None
        next_layer_size = parent1[i+1]['layer_size'] if i < len(parent1)-1 else None
        next_kernel_size = parent1[i+1]['kernel_size'] if i < len(parent1)-1 else None
        
        if prev_layer_size and layer['kernel_size'] > prev_layer_size:
            layer['kernel_size'] = prev_layer_size
        
        if prev_kernel_size and layer['kernel_size'] > prev_kernel_size:
            layer['kernel_size'] = prev_kernel_size
            
        if next_layer_size and layer['kernel_size'] > next_layer_size:
            layer['kernel_size'] = next_layer_size
            
        if next_kernel_size and layer['kernel_size'] > next_kernel_size:
            layer['kernel_size'] = next_kernel_size
        
        # Append the selected layer to the child chromosome
        child.append(layer)
    
    return child

def convert_chromosome_to_nn(chromosome, input_shape, output_shape):
    input_channels = 1
    layers = []
    input_size, = input_shape
    output_size, = output_shape
    for layer_info in chromosome:
        layer_size = layer_info['layer_size']
        kernel_size = layer_info['kernel_size']
        dropout = layer_info['dropout']
        activation_str = layer_info['activation']

        activation = getattr(nn, activation_str)()

        # Skip the current layer if the kernel size is larger than the input size
        if kernel_size > input_size:
            continue

        conv = nn.Conv1d(input_channels, layer_size, kernel_size)
        layers.extend([conv, activation])

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        input_channels = layer_size

    # Calculate the size of the flattened output after passing through convolutional layers
    dummy_input = torch.randn(1, input_size).unsqueeze(0)
    conv_output = nn.Sequential(*layers)(dummy_input)
    flattened_size = conv_output.view(conv_output.size(0), -1).size(1)

    layers.append(nn.Flatten())
    layers.append(nn.Linear(flattened_size, output_shape[0]))
    layers.append(nn.Sigmoid())  # Add Sigmoid activation function to the output layer

    return nn.Sequential(*layers)

def train_nn(nn_model: nn.Module, train_loader: DataLoader, device: torch.device, epochs : int,
            criterion, optimizer) -> float:

    nn_model = nn_model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for step ,( inputs,targets)  in  enumerate(tqdm.tqdm(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            #print("the inputs ",inputs, inputs.shape)
            #print("the outpus:",targets,targets.shape)
            optimizer.zero_grad()
            outputs = nn_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f" Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_loader)}")

    return running_loss / (epoch + 1)

def evaluate_nn(nn_model: nn.Module, test_loader: DataLoader, device: torch.device,criterion) -> float:
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = nn_model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
        return total_loss / len(test_loader)

def plot_graphs(stats):
    for gen_stats in stats:
        gen = gen_stats['Gen']
        loss = gen_stats['loss_of_gen']
        fitness = gen_stats['fitness_of_gen']
        
        plt.figure(figsize=(12, 6))
        plt.title(f"Generation {gen}")
        plt.plot(loss, label='Loss')
        plt.plot(fitness, label='Fitness')
        plt.xlabel('Individual')
        plt.ylabel('Value')
        plt.legend()
        plt.show()



def plot_fitness(stats):
    generations = [stat['Gen'] for stat in stats]
    fitness = [stat['fitness_of_gen'] for stat in stats]
    
    fig, ax = plt.subplots()
    ax.bar(generations, [np.mean(f) for f in fitness], yerr=[np.std(f) for f in fitness])
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Over Generations')
    plt.show()



def new_population(population, fitness, num_parents, mutation_rate, layer_sizes, kernel_sizes, dropouts, activations, input_shape, output_shape):
    new_population = []

    # Select the best performing individuals as parents for the next generation
    parents = [population[i] for i, score in sorted(enumerate(fitness), key=lambda x: x[1])[:num_parents]]

    # Create the children from the selected parents
    for i in range(len(population) - num_parents):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = crossover(parent1, parent2)
        child = mutate_chromosome(child, layer_sizes, kernel_sizes, dropouts, activations, mutation_rate, input_shape[0])
        new_population.append(child)

    # Add the parents to the new population
    new_population.extend(parents)

    return new_population



if __name__ == "__main__":
    data_folder = config['data_folder']
    subjects_range = config['subjects_range']
    series_range = config['series_range']
    train_test_split_ratio = 0.8
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

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    #device = 'cpu'

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
            fitness_of_gen.append(fittness)
            loss_of_gen.append(loss)
            print(fittness)
            count_individual += 1

            """
            # Mutate the chromosome with a mutation_rate of 1
            mutated_individual = mutate_chromosome(individual, layer_sizes, kernel_size, drop_out, activations, mutation_rate=1, input_size=input_size)
            print("Parent2:")
            for mutated_layer in mutated_individual:
                print(" ", mutated_layer)
            nn_model = convert_chromosome_to_nn(mutated_individual, input_shape=(input_size,), output_shape=(output_size,))
            optimizer = optim.Adam(nn_model.parameters(),learning_rate)
            optimizer.zero_grad()
            outputs = nn_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            count_individual += 1
            offspring1 = crossover(individual,mutated_individual,0.5)
            print("Offspring")
            for layer in offspring1:
                print(" ", layer)
            nn_model = convert_chromosome_to_nn(offspring1, input_shape=(input_size,), output_shape=(output_size,))
            optimizer = optim.Adam(nn_model.parameters(),learning_rate)
            optimizer.zero_grad()
            outputs = nn_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            """
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

        #train_nn(nn_model=nn_model, train_loader=test_loader, epochs=1, device=device,optimizer=optimizer,criterion=criterion)
        
        #for i in range(len(individual)):
        #    print("l: ",individual[i],"m: ", mutated_individual[i],"o:", offspring1[i])
        
        #fitness = evaluate_nn(nn_model, test_loader, device,criterion)

    print("STATS:", stats)
    plot_fitness(stats=stats)
    #plot_graphs(stats)


