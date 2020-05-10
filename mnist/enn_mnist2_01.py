import pandas as pd
from collections import defaultdict
import warnings
import visualize
import neat
import multiprocessing
import os
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
import numpy as np

X, y = loadlocal_mnist(
    images_path='c:/ENN/mnist/train-images-idx3-ubyte',
    labels_path='c:/ENN/mnist/train-labels-idx1-ubyte')

number_of_classification = 2
number_of_sampling = 100

list_y = y.tolist()
digits_indexes = []
for digit in range(number_of_classification):
    li = [i for i in range(len(list_y)) if list_y[i] == digit]
    digits_indexes.extend(li[:number_of_sampling])

samplesize = X[digits_indexes].shape[0]
x_inputs =[tuple(c) for c in X[digits_indexes].tolist()]
x_outputs = [tuple(c) for c in y[digits_indexes].reshape(samplesize,1).tolist()]

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        outputs = []
        for xi in x_inputs:
            output = net.activate(xi)
            outputs.append(output)
        
    
        px_outputs = np.array(outputs).reshape(samplesize, 1)
        px_outputs[px_outputs>0] = 1
        px_outputs[px_outputs<=0] = 0
        acc = np.sum(px_outputs == np.array(x_outputs))/samplesize
#         px_outputs = softmax(np.array(outputs).reshape(samplesize, 10), axis=1)
#         #mse = np.mean(np.sum(np.square(px_outputs - np.array(x_outputs)), axis = 1))
#         px_outputs[px_outputs < np.exp(-100)] = np.exp(-100)
#         ce = np.mean(np.sum(np.array(x_outputs) * np.log(px_outputs), axis = 1))

        genome.fitness = acc

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

    # add a stdout reporter to show progress in the terminal
    reporter = neat.StdOutReporter(False)
    p.add_reporter(reporter)
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #checkpointer = neat.Checkpointer(100)
    #p.add_reporter(checkpointer)
    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 10)

    return [stats, winner]


samplesize = len(x_outputs)

local_dir = os.getcwd()
config_path = os.path.join(local_dir, "config-feedforward-mnist2")
[stats, winner] = run(config_path)

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

correct = 0        
for xi,xo in zip(x_inputs,x_outputs):
    output = winner_net.activate(xi)
    xo = np.array(xo).tolist()
    R = xo[0]
    if output[0] > 0:
        P = 1
    else:
        P = 0
    if R == P:
        correct += 1
    print("R:{} P:{}".format(R, P))

print("Accuracy: {}".format(correct/samplesize))

visualize.draw_net(config, winner, True, fmt="png")
visualize.plot_stats(stats, ylog=False, view=False)
visualize.plot_species(stats, view=False)

