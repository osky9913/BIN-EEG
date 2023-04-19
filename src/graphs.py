import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from matplotlib.ticker import FuncFormatter

# Data from your provided stats

with open('results.pickle', 'rb') as f:

    stats = pickle.load(f) # deserialize using load()
    #print(stats)

                                                            

def box_plot_mat(stats):
    fitness_data = [[1 - fitness for fitness in stat['fitness_of_gen']]
                for stat in stats]
    num_generations = len(fitness_data)
    x_ticks = [x+1 for x in range(num_generations)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(fitness_data)

    print([x for x in range(num_generations)])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    plt.suptitle('Fitness Distribution per Generation (1 - Fitness)')
    plt.show()


def box_plot(stats):
    fitness_data = [[1 - fitness for fitness in stat['fitness_of_gen']]
                for stat in stats]
    num_generations = len(fitness_data)
    x_ticks = [x for x in range(num_generations)]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=fitness_data, ax=ax)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    plt.suptitle('Fitness Distribution per Generation (1 - Fitness)')
    plt.show()

def log_tick_labels(y, pos):
    return '{:.3f}'.format(100**y)

def bar_graph_mat(stats):
    data = []
    count_indvidual = 0
    for stat in stats:
        gen = stat["Gen"]
        loss_of_gen = stat["loss_of_gen"]
        fitness_of_gen = stat["fitness_of_gen"]
        len(fitness_of_gen)

        for index, (loss, fitness) in enumerate(zip(loss_of_gen, fitness_of_gen)):
            data.append({"Gen": "gen: " +str(gen+1), "Loss": loss, "Fitness": fitness, "Individual": count_indvidual})
            count_indvidual+=1

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(6, 4))


    df['Fitness'] = df['Fitness'].apply(lambda x:1 - x )
    ax.bar( df["Individual"],df["Fitness"], color='C1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    #ax.set_ylim(0, 1)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_labels))

    ax.margins(0.05)
    plt.show()

    print(df)


def catplot(stats):
    data = []
    for stat in stats:
        gen = stat["Gen"]
        loss_of_gen = stat["loss_of_gen"]
        fitness_of_gen = stat["fitness_of_gen"]
        len(fitness_of_gen)
        for index, (loss, fitness) in enumerate(zip(loss_of_gen, fitness_of_gen)):
            data.append({"Gen": "gen: " +str(gen+1), "Loss": loss, "Fitness": fitness, "Individual": str(gen+1)+ ":" +str(index+1)})

    df = pd.DataFrame(data)
    print(df)


    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    custom_palette_two = [ colors[int(((str(ind).split(":"))[0]))] for ind in  df["Individual"]]
    sns.catplot(data=df, x="Gen", y="Fitness", hue="Individual", palette=custom_palette_two,edgecolor='grey', kind='bar')
    plt.title("Fitness per Generation")
    plt.show()


#catplot(stats)
#box_plot(stats)

box_plot_mat(stats=stats)
bar_graph_mat(stats=stats)