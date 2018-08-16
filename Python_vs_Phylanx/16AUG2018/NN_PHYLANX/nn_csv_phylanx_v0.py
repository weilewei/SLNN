from phylanx.ast import *

import pandas as pd
import numpy as np
import time
import argparse
import sys

if not len(sys.argv) == 7:
    print("This program requires the following 6 arguments seperated by a space ")
    print("iterations Row_START Row_STOP Col_START Col_STOP hiddenlayer_neurons")
    exit(-57)

parser = argparse.ArgumentParser(description='Iteration and slicing operations')
parser.add_argument('integers', metavar='Iteration  + Slicing Parameters', type=int, nargs='+',
                    help='iterations, Row_START, Row_STOP, Col_START, Col_STOP, hiddenlayer_neurons')

args = parser.parse_args()
print("Command Line: ", args.integers[0], args.integers[1], args.integers[2], args.integers[3], args.integers[4],
      args.integers[5])

iterations = args.integers[0]
row_start = args.integers[1]
row_stop = args.integers[2]
col_start = args.integers[3]
col_stop = args.integers[4]
hiddenlayer_neurons = args.integers[5]

treading = time.time()

print("Reading Data ....")
df = pd.read_csv('/phylanx-data/CSV/10kx10k.csv', sep=',', header=None)
df = df.values
print("Slicing ....")
X = df[row_start:row_stop, col_start:col_stop]
Y = np.squeeze(np.asarray(df[row_start:row_stop, 10000:10001]))
trslice = time.time()
print("Reading and Slicing done in ", trslice - treading, " s ")

print("Starting SLNN ....")

tnn = time.time()


@Phylanx
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@Phylanx
def derivatives_sigmoid(x):
    return x * (1 - x)


@Phylanx
def SingleLayerNeuralNetwork(x, y, hiddenlayer_neurons, iterations, lr):
    inputlayer_neurons = np.shape(x)[1]  # number of features in data set
    output_neurons = np.shape(y)[0]  # number of neurons at output layer

    # weight and bias initialization
    wh = random([inputlayer_neurons, hiddenlayer_neurons], "uniform")
    bh = random([1, hiddenlayer_neurons], "uniform")
    wout = random([hiddenlayer_neurons, output_neurons], "uniform")
    bout = random([1, output_neurons], "uniform")

    # for i in range(iterations):
    i = 0
    while (i < iterations):
        # Forward Propogation
        hidden_layer_input1 = np.dot(x, wh)
        hidden_layer_input = hidden_layer_input1 + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1 = np.dot(hiddenlayer_activations, wout)
        output_layer_input = output_layer_input1 + bout
        output = sigmoid(output_layer_input)

        # Backpropagation
        E = y - output
        slope_output_layer = derivatives_sigmoid(output)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        d_output = E * slope_output_layer
        Error_at_hidden_layer = np.dot(d_output, np.transpose(wout))
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        wout += (np.dot(np.transpose(hiddenlayer_activations), d_output)) * lr
        bout += np.sum(d_output, 0, True) * lr
        wh += (np.dot(np.transpose(x), d_hiddenlayer)) * lr
        bh += np.sum(d_hiddenlayer, 0, True) * lr
        i += 1

    return wh


weights = SingleLayerNeuralNetwork(X, Y, hiddenlayer_neurons, iterations, 1e-5)

tfinal = time.time()

print(" result = ", weights, " in time =", tfinal - tnn)
