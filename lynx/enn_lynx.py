import pandas as pd
import numpy as np


inputs = "7"

df=pd.read_csv('csvs/' + inputs + 'lynx_train_X.csv', sep=' ',header=None)
new_train_X = df.values
df=pd.read_csv('csvs/' +inputs + 'lynx_train_Y.csv', sep=' ',header=None)
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
    if_no_connect = True

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    error = 0.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        error -= (output[0] - xo[0]) ** 2
        # error -= np.abs(output[0] - xo[0])
        if float(output[0]) != 0:
            if_no_connect = False
    
    if if_no_connect:
        mse = -1  
    # there is no connection at all
    else:
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

    # Display the winning genome.
    print("\nBest genome:\n{!s}".format(winner))

    # Show output of the most fit genome against training data.
    print("\nOutput:")
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
    
    
    df=pd.read_csv('csvs/' +inputs + 'lynx_test_X.csv', sep=' ',header=None)
    new_test_X = df.values
    X_test_inputs = []
    for i in range(len(new_test_X)):
        X_test_inputs.append(tuple(new_test_X[i]))

    predictions_enn = []
    for xi in X_test_inputs:
        output = winner_net.activate(xi)
        predictions_enn.append(output)

    np.savetxt('csvs/' +inputs + 'predictions_enn.csv', np.array(predictions_enn), delimiter=',')
    real_y=pd.read_csv('csvs/' +inputs + 'lynx_test_Y.csv', sep=' ',header=None)
    mse = np.sum((np.array(real_y) - predictions_enn)**2)/(len(predictions_enn))
    mae = np.average(np.abs(np.array(real_y) - predictions_enn))
    print("MSE:", mse)
    print("MAE:", mae)

    node_names = {-1: "t-7", -2: "t-6", -3: "t-5", -4:"t-4", -5:"t-3", -6:"t-2", -7:"t-1",0: "Target"}
    visualize.draw_net(config, winner, True, node_names=node_names,fmt="png")
    visualize.plot_stats(stats, ylog=False, view=False)
    visualize.plot_species(stats, view=False)





if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.getcwd()
    config_path = os.path.join(local_dir, "config-feedforward")
    run(config_path)