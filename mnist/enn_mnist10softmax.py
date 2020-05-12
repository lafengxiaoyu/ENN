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


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

X, y = loadlocal_mnist(
    images_path='c:/ENN/mnist/train-images-idx3-ubyte',
    labels_path='c:/ENN/mnist/train-labels-idx1-ubyte')

numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # number 2 > number 1
number_of_classification = len(numbers)
number_of_sampling = 100

list_y = y.tolist()


import random
digits_indexes = []
for digit in numbers:
    li = [i for i in range(len(list_y)) if list_y[i] == digit]
    digits_indexes.extend(random.sample(li, number_of_sampling))    
    
samplesize = X[digits_indexes].shape[0]
x_inputs =[tuple(c) for c in X[digits_indexes].tolist()]
x_outputs = [tuple(c) for c in y[digits_indexes].reshape(samplesize,1).tolist()]




def softmax(x, axis=1):
    # 计算每行的最大值
    row_max = x.max(axis=axis)
 
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max=row_max.reshape(-1, 1)
    x = x - row_max
 
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        outputs = []
        for xi in x_inputs:
            output = net.activate(xi)
            outputs.append(output)
        
        
        px_outputs = softmax(np.array(outputs).reshape(samplesize, number_of_classification), axis=1)
        # the index of maximum in each line
        pred_outputs = np.argmax(px_outputs, axis = 1)
        # replace index to real number, from higher number to lower number!!
        pred_outputs[pred_outputs==2] = numbers[2]
        pred_outputs[pred_outputs==1] = numbers[1]
        pred_outputs[pred_outputs==0] = numbers[0]
        
        real_outputs = np.array(x_outputs).reshape(samplesize,)
        
        acc = np.sum(pred_outputs == real_outputs)/samplesize

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
    winner = p.run(eval_genomes, 2)

    return [stats, winner]


samplesize = len(x_outputs)

local_dir = os.getcwd()
config_path = os.path.join(local_dir, "config-feedforward-mnist10")
[stats, winner] = run(config_path)

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

number_of_test = 100

import random
digits_indexes = []
for digit in numbers:
    li = [i for i in range(len(list_y)) if list_y[i] == digit]
    digits_indexes.extend(random.sample(li, number_of_test))

testsamplesize = X[digits_indexes].shape[0]
test_x_inputs =[tuple(c) for c in X[digits_indexes].tolist()]
test_x_outputs = [tuple(c) for c in y[digits_indexes].reshape(testsamplesize,1).tolist()]


outputs = []
for xi in test_x_inputs:
    output = winner_net.activate(xi)
    outputs.append(output)

px_outputs = softmax(np.array(outputs).reshape(testsamplesize, number_of_classification), axis=1)
# the index of maximum in each line
pred_outputs = np.argmax(px_outputs, axis = 1)
# replace index to real number, from higher number to lower number!!
for i in range(number_of_classification):
    number = numbers[number_of_classification - i - 1]
    pred_outputs[pred_outputs==number] = numbers[number]


real_outputs = np.array(test_x_outputs).reshape(testsamplesize,)

acc = np.sum(pred_outputs == real_outputs)/testsamplesize
print("Accuracy: {}".format(acc))

visualize.draw_net(config, winner, True, fmt="png")
visualize.plot_stats(stats, ylog=False, view=False)
visualize.plot_species(stats, view=False)

