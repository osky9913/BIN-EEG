import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from matplotlib.ticker import FuncFormatter
import pprint

# Data from your provided stats

with open('results.pickle', 'rb') as f:

    stats = pickle.load(f) # deserialize using load()
    #print(stats)

                                                        
def plot_box_accuracy(stats):
    generations = []
    accuracies = []
    fig, ax = plt.subplots() 

    for gen in stats:
        gen_accuracies = [ind['accuraccy'] for ind in stats[gen].values()]
        generations.append(gen)
        accuracies.append(gen_accuracies)

    ax.boxplot(accuracies, labels=generations)
    ax.set_xlabel("Generations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Box plot of accuracy per generation")
    fig.savefig('Box_plot_of_accuracy_per_generation.png')

    #plt.show()

def plot_loss_lines(stats):
    fig, ax = plt.subplots() 
    for gen in stats:
        for ind in stats[gen]:
            losses = stats[gen][ind]['losses']
            ax.plot(losses, label=f"{gen}-{ind}")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Line graph of losses per individual")
    #plt.legend()
    fig.savefig('Line_graph_of_losses_per_individual.png')

    #plt.show()

def average(numbers_list):
    total = sum(numbers_list)
    count = len(numbers_list)
    return total / count

def plot_accuracy_per_gen(stats):
    gen_values = []
    for gen in stats:
        gen_accuracies = average([ind['accuraccy'] for ind in stats[gen].values()])
        gen_values.append(gen_accuracies)

    fig, ax = plt.subplots()

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax.plot(gen_values, label=gen)
    ax.set_xlabel("Individuals")
    ax.set_ylabel(" ")
    ax.set_title("Accuracy mean per generation")
    ax.legend("")
    
    fig.savefig('Accuracy_mean_per_generationl.png')

    #plt.show()

def plot_accuracy_individuals(stats):
    individuals = []
    accuracies = []
    fig, ax = plt.subplots()
    individual = 0 
    for gen in stats:
        for ind in stats[gen]:

            individuals.append(f"{ind}")
            accuracies.append(stats[gen][ind]['accuraccy'])
    plt.rcParams["figure.autolayout"] = True
    x = np.arange(0, len(accuracies), 1)

    ax.plot(x, accuracies)
    ax.set_xlabel("Individuals")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per individuals")
    ax.set_xticks(np.arange(min(x), max(x)+1, 1))  # Set x-axis ticks from 0 to 10 with a step of 1

    #ax.set_xticks(rotation=45)
    fig.savefig('Accuracy_per_individuals.png')

    #plt.show()

def find_best_accuracy(stats):
    best_accuracy = float('-inf')
    best_individual = None

    for generation, individuals in stats.items():
        for individual_id, individual_data in individuals.items():
            accuracy = individual_data['accuraccy']
            if accuracy >best_accuracy:
                best_accuracy = accuracy
                best_individual = (generation, individual_id)
    print(best_individual, best_accuracy)
    return best_individual, best_accuracy

best_individual, best_accuracy = find_best_accuracy(stats)
print(f"The best individual is {best_individual} with an accuracy of {best_accuracy}.")


find_best_accuracy(stats=stats)
plot_box_accuracy(stats)
plot_loss_lines(stats)
plot_accuracy_per_gen(stats)
plot_accuracy_individuals(stats)

pprint.pprint(stats)
#catplot(stats)
#box_plot(stats)

#box_plot_mat(stats=stats)
#bar_graph_mat(stats=stats)