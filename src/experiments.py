import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import pprint
import os
import json
import tqdm

from eegDataset import EegDataset
import config


config = config.config



def create_children_from_worse_half(population):
    children  = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            parent1 = population[i]
            parent2 = population[j]

            child1, child2 = crossover(parent1, parent2)
            children.append(child1)
            children.append(child2)


    return children


def create_population(num_individuals, input_size, output_size, min_layers, max_layers, min_nodes, max_nodes, max_pooling_layers):
    population = []
    for _ in range(num_individuals):
        num_layers = random.randint(min_layers, max_layers)
        layers = []
        pool_count = 0
        for _ in range(num_layers):
            if pool_count < max_pooling_layers:
                layer_type = random.choice(["conv", "pool"])
            else:
                layer_type = "conv"

            if layer_type == "conv":
                num_nodes = random.randint(min_nodes, max_nodes)
                layers.append({"type": layer_type, "nodes": num_nodes, "activation": "relu"})
            else:
                pool_count += 1
                pool_type = random.choice(["max", "avg"])
                layers.append({"type": layer_type, "pool": pool_type})
        layers.append({"type": "linear", "nodes": output_size, "activation": "softmax"})
        chromosome = {"input_size": input_size, "layers": layers , "parents": []}
        population.append(chromosome)
    return population


def get_layers_by_type(layers):
    conv_layers = [layer for layer in layers if layer['type'] == 'conv']
    pool_layers = [layer for layer in layers if layer['type'] == 'pool']
    linear_layers = [layer for layer in layers if layer['type'] == 'linear']
    return conv_layers, pool_layers, linear_layers





def mutate(chromosome, mutation_rate, min_nodes, max_nodes):
    chromosome["old_layers"] = chromosome["layers"]
    for layer in chromosome["layers"]:
        if random.random() < mutation_rate:
            if layer["type"] == "conv":
                layer["nodes"] = random.randint(min_nodes, max_nodes)
            else:
                layer["pool"] = random.choice(["max", "avg"])
    return chromosome

def build_model(chromosome, input_size, output_size):
    layers = []
    in_channels = 1
    for layer_info in chromosome["layers"]:
        if layer_info["type"] == "conv":
            layers.append(nn.Conv1d(in_channels, layer_info["nodes"], kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = layer_info["nodes"]
        elif layer_info["type"] == "pool":
            if layer_info["pool"] == "max":
                layers.append(nn.MaxPool1d(kernel_size=2))
            else:
                layers.append(nn.AvgPool1d(kernel_size=2))
            input_size = input_size // 2
        elif layer_info["type"] == "linear":
            layers.append(nn.Flatten())
            layers.append(nn.Linear(in_channels * input_size, output_size))
    layers.append(nn.Sigmoid())

            # Removed Softmax layer
    return nn.Sequential(*layers)



def train_nn(model, train_loader, epochs, learning_rate,device ):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for step ,( inputs,targets)  in  enumerate(tqdm.tqdm(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f" Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_loader)}")
        losses.append((running_loss / len(train_loader)))
    return losses



def evaluate_accuracy(model, dataloader, device, individual):
    correct = 0
    total = 0
    results = []



    with torch.no_grad():
        for step, (inputs, targets) in enumerate(tqdm.tqdm(dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = (outputs.data > 0.5).float()  # Apply a threshold (0.5) to the outputs
            total += targets.size(0)
            correct += ((predicted == targets).sum(dim=1) == targets.size(1)).sum().item()

            for i in range(len(targets)):
                example = {
                    "targets": targets[i].tolist(),
                    "predicted": predicted[i].tolist(),
                    "outputs": outputs[i].tolist(),
                }
                results.append(example)

    accuracy = correct / total



    return accuracy, results


def select_top_k_indices(scores, k):
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

def select_bottom_k_indices(scores, k):
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)[:k]

#def train_evolution(train_loader, test_loader,population, epochs, mutation_rate, num_generations, train_epochs, learning_rate):

def create_check(population, generation ,device, test_loader):
    inputs, targets = next(iter(test_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    print(inputs)
    print(targets)

    for generation in range(num_generations):
        print(f"Generation Check {generation + 1}")
        fitness_scores = []
        count_chromo = 0
        stats[f"Generation {generation + 1}"] = {}
        for chromosome in population:
            stats[f"Generation {generation + 1}"]["individual" + str(count_chromo)] ={}
            stats[f"Generation {generation + 1}"]["individual" + str(count_chromo)]["chromosome"] = chromosome
            print(f"Generation Check {generation + 1} { count_chromo}")
            pprint.pprint(chromosome)
            model = build_model(chromosome, input_size, output_size)
            model.to(device=device)
            criterion = nn.BCEWithLogitsLoss()  # Changed from BCELoss to BCEWithLogitsLoss
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            #losses = []
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()            
            accuracy = random.uniform(0.1, 0.5)
            #print(accuracy)
            fitness_scores.append(accuracy)
            count_chromo += 1



        #pprint.pprint(stats)
        selected_top_indices = select_top_k_indices(fitness_scores, k=int(len(population) * 0.5))
        better_population = [population[i] for i in selected_top_indices]
        new_population = copy.deepcopy(better_population)

        selected_bottom_indices = select_bottom_k_indices(fitness_scores, k=int(len(population) * 0.5))
        worse_population = [population[i] for i in selected_bottom_indices]


        print("Better population", len(new_population))
        print("Worse population" , len(worse_population))
        #pprint.pprint(worse_population)
        #print(new_population)
        if len(new_population) < len(population):
            
            better_count = 0
            worse_count = 0 
            child_count = 0
            #childrend = create_children_from_worse_half(population)

            while len(new_population) < len(population):
                #if len(childrend) == 0:
                #    print("no children")
                #    exit(1)
                    #--------------------------
                #if child_count < len(childrend):
                 #   childrend
                #    new_population.append(mutate(childrend[child_count], mutation_rate, min_nodes, max_nodes))
                #    child_count +=1
               # else:
                new_population.append(mutate(better_population[better_count], mutation_rate, min_nodes, max_nodes))
                new_population.append(mutate(worse_population[worse_count], mutation_rate, min_nodes, max_nodes))
                worse_count += 1 
                better_count+=1
            print("better_count ", better_count)
            print("worse_count ", worse_count)
            print("child_count ", child_count)
            population = new_population


# Set up the parameters for the genetic algorithm

if __name__ == '__main__':
    num_individuals = config['num_individuals']
    input_size = 32
    output_size = 6
    min_layers = config['min_layers']
    max_layers = config['max_layers']
    min_nodes = 16
    max_nodes = 64
    epochs = config['epochs']
    mutation_rate = 0.5
    num_generations = 2
    learning_rate = config['learning_rate']
    data_folder = config['data_folder']
    subjects_range = config['subjects_range']
    series_range = config['series_range']
    train_test_split_ratio = config['train_test_split_ratio']
    batch_size = 2048
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    #device = 'cpu'

    max_pooling_layers = 0
    temp_input_size = input_size
    while temp_input_size > 1:
        temp_input_size //= 2
        max_pooling_layers += 1

    # Generate the initial population

    # Generate the initial population
    population = create_population(num_individuals, input_size, output_size, min_layers, max_layers, min_nodes, max_nodes, max_pooling_layers)
    stats ={}

    train_data = EegDataset(data_folder, subjects_range, series_range, train_test_split_ratio, train=True)
    test_data = EegDataset(data_folder, subjects_range, series_range, train_test_split_ratio, train=False)


    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,num_workers=4)

    create_check(population=population,generation=num_generations,device=device,test_loader=test_loader)
    individual = 0
    for generation in range(num_generations):
        print(f"Generation Check {generation + 1}")
        fitness_scores = []
        count_chromo = 0
        stats[f"Generation {generation + 1}"] = {}
        for chromosome in population:
            stats[f"Generation {generation + 1}"]["individual" + str(count_chromo)] ={}
            stats[f"Generation {generation + 1}"]["individual" + str(count_chromo)]["chromosome"] = chromosome
            print(count_chromo)
            print(chromosome)
            model = build_model(chromosome, input_size, output_size)
            model.to(device=device)
            losses = train_nn(model, train_loader, epochs, learning_rate,device)
            accuracy,results = evaluate_accuracy(model, test_loader,device,individual=individual)
            filename = "validationsIndividuals/" + str(individual) + ".txt"
            if not os.path.exists(filename):
                with open(filename, "w") as f:
                    json.dumps(results,f)
            f.close()
            stats[f"Generation {generation + 1}"]["individual" + str(count_chromo)]["losses"] = losses
            stats[f"Generation {generation + 1}"]["individual" + str(count_chromo)]["accuraccy"] = accuracy

            print(accuracy)
            fitness_scores.append(accuracy)
            count_chromo += 1
            individual+=1

        pprint.pprint(stats)
        selected_top_indices = select_top_k_indices(fitness_scores, k=int(len(population) * 0.5))
        better_population = [population[i] for i in selected_top_indices]
        new_population = copy.deepcopy(better_population)

        selected_bottom_indices = select_bottom_k_indices(fitness_scores, k=int(len(population) * 0.5))
        worse_population = [population[i] for i in selected_bottom_indices]

        if len(new_population) < len(population):
            better_count = 0
            worse_count = 0 
            child_count = 0
            while len(new_population) < len(population):
                new_population.append(mutate(better_population[better_count], mutation_rate, min_nodes, max_nodes))
                better_count+=1

            print("better_count ", better_count)

            population = new_population

    best_chromosome = population[0] # Assumes the first chromosome in the population is the best one
    best_model = build_model(best_chromosome, input_size, output_size)
    print(best_chromosome)
    with open('results.pickle', 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
