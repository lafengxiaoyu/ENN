import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
import os
import neat
import visualize

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


import warnings
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


d = 2
samplesize = 1000
runs = 2000

def ackley(a):
    d = len(a)
    term1 = - 20 * np.exp(-0.2*np.sqrt(np.sum(a**2)/d))
    term2 = - np.exp(np.sum(np.cos(a*2*np.pi))/d)
    f = term1 + term2 + 20 + np.exp(1)
    return f


def sampleing():
    x_inputs = []
    x_outputs = []
    list_outputs = []

    for i in range(samplesize):
        A = -32.768
        B = 32.768#小数的范围A ~ B
        a = np.random.uniform(A,B,d)
        x_inputs.append(tuple(a/A))
        y = ackley(a)
        list_outputs.append(y)

    for i in range(samplesize):
        # x_outputs.append(tuple([list_outputs[i]/np.max(list_outputs)]))
        x_outputs.append(tuple([list_outputs[i]/24]))
    return [x_inputs, x_outputs]

[x_inputs, x_outputs] = sampleing()

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
    for xi, xo in zip(x_inputs, x_outputs):
        output = net.activate(xi)
        error -= (output[0] - xo[0]) ** 2
        # error -= np.abs(output[0] - xo[0])
        if float(output[0]) != 0:
            if_no_connect = False
    
    if if_no_connect:
        mse = -1
    else:
        mse = error/samplesize
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
    winner = p.run(pe.evaluate, runs)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    outputs = []
    for xi, xo in zip(x_inputs, x_outputs):
        output = winner_net.activate(xi)
        outputs.append(output)
        print("input {!r}, expected output {!r}, got {!r}".format(
            xi, xo, output))

    xi = tuple([0,0])
    xo = [tuple([0])]
    output = winner_net.activate(xi)
    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


    fig = plt.figure(figsize = (16,9))
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(-32.768, 32.768, 0.25)
    Y = np.arange(-32.768, 32.768, 0.25)
    X, Y = np.meshgrid(X, Y)

    OZ = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            a = np.array([X[i][j]/32.768,Y[i][j]/32.768])
            OZ.append(winner_net.activate(tuple(a)))
    OZ = np.array(OZ).reshape(X.shape)

    Z = OZ*24
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha = 0.4)

    # Customize the z axis.
    ax.set_zlim(-1.01, 25)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    outputs = []
    for xi in x_inputs:
        output = winner_net.activate(xi)
        outputs.append(output)

    x = []
    y = []
    z = []
    for i in range(1000):
        x.append(x_inputs[i][0]*32.768)
        y.append(x_inputs[i][1]*32.768)
        z.append(outputs[i][0]*24)

    ax.scatter(x, y, z, c='k', marker='o')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("fitting_ackley2")

    return winner




if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    

    local_dir = os.getcwd()
    config_path = os.path.join(local_dir, "config-feedforward2")
    winner = run(config_path)

