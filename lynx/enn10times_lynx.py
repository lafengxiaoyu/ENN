import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

inputs = "7"

df=pd.read_csv('csvs/' + inputs + 'lynx_train_X.csv', sep=' ',header=None)
new_train_X = df.values
df=pd.read_csv('csvs/' + inputs + 'lynx_train_Y.csv', sep=' ',header=None)
new_train_Y = df.values

X_train_inputs = []
Y_train_outputs = []

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# L = len(new_train_X)
L = len(new_train_X)
for i in range(L):
    X_train_inputs.append(tuple(new_train_X[i]))
for i in range(L):
    Y_train_outputs.append(tuple(new_train_Y[i]))

import multiprocessing
import os
import neat
import visualize

import warnings

xor_inputs = X_train_inputs
xor_outputs = Y_train_outputs

def eval_genome(genome, config):
    """
    This function will be run in parallel by ParallelEvaluator.  It takes two
    arguments (a single genome and the genome class configuration data) and
    should return one float (that genome's fitness).

    Note that this function needs to be in module scope for multiprocessing.Pool
    (which is what ParallelEvaluator uses) to find it.  Because of this, make
    sure you check for __main__ before executing any code (as we do here in the
    last few lines in the file), otherwise you'll have made a fork bomb
    instead of a neuroevolution demo. :)
    """

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    error = 0.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        error -= (output[0] - xo[0]) ** 2
        # error -= np.abs(output[0] - xo[0])
    mse = error/L
    # mad = error/L
    return mse


def run(config_file):
    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, 2000)

    return stats





if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.getcwd()
    config_path = os.path.join(local_dir, "config-feedforward")
    
    stats_list = []
    for i in range(10):
        stats = run(config_path)
        stats_list.append(stats)
        

    avg_fitness_list = []
    best_list = []

    for i in range(10):
        generation = range(len(stats_list[i].get_fitness_mean()))
        avg_fitness = np.array(stats_list[i].get_fitness_mean())
        avg_fitness_list.append(avg_fitness)
        best_fitness = [c.fitness for c in stats_list[i].most_fit_genomes]
        best_list.append(best_fitness)
        
        plt.plot(generation, avg_fitness, alpha = 0.4)
        plt.plot(generation, best_fitness, alpha = 0.4)

    plt.plot(generation, np.average(np.array(avg_fitness_list),axis = 0), label="Average")
    plt.plot(generation, np.average(np.array(best_list),axis = 0), label="Best")


    plt.savefig('average10times.png')