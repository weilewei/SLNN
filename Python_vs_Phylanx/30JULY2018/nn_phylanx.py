from phylanx.ast import *
import numpy as np
import time


@Phylanx
def main_phylanx(X, y, wh, bh, wout, bout, lr, num_iter):
    # create local variable
    wh_local = wh
    bh_local = bh
    wout_local = wout
    bout_local = bout
    output = 0
    hidden_layer_input1 = 0
    hidden_layer_input = 0
    hiddenlayer_activations = 0
    output_layer_input1 = 0
    output_layer_input = 0
    slope_hidden_layer = 0
    slope_output_layer = 0
    E = 0
    d_output = 0
    Error_at_hidden_layer = 0
    d_hiddenlayer = 0

    for i in range(num_iter):
        # forward
        hidden_layer_input1 = np.dot(X, wh_local)
        hidden_layer_input = hidden_layer_input1 + bh_local
        hiddenlayer_activations = (1 / (1 + np.exp(-hidden_layer_input)))
        output_layer_input1 = np.dot(hiddenlayer_activations, wout_local)
        output_layer_input = output_layer_input1 + bout_local
        output = (1 / (1 + np.exp(-output_layer_input)))

        # Backpropagation
        E = y - output
        # print("E", E)
        slope_output_layer = (output * (1 - output))
        slope_hidden_layer = (hiddenlayer_activations * (1 - hiddenlayer_activations))
        d_output = E * slope_output_layer
        Error_at_hidden_layer = np.dot(d_output, np.transpose(wout_local))  # some problem
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        wout_local += (np.dot(np.transpose(hiddenlayer_activations), d_output)) * lr
        bout_local += np.sum(d_output, 0, True) * lr
        # bout_local += np.sum(d_output, axis=0, keepdims=True) * lr
        wh_local += (np.dot(np.transpose(X), d_hiddenlayer)) * lr
        bh_local += np.sum(d_hiddenlayer, 0, True) * lr
        # bh_local += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr
   # print(np.shape(bout_local))
    return output


# Variable initialization
# Input array
a = 3
b = 4
c = 3 
X = np.ones((a, b))

# Output
output_y = np.ones((1, a))

num_iter = 5000 # Setting training iterations
lr = 0.1  # Setting learning rate
inputlayer_neurons = X.shape[1]  # number of features in data set
hiddenlayer_neurons = c  # number of hidden layers neurons
output_neurons = a  # number of neurons at output layer

# weight and bias initialization
wh = np.ones((inputlayer_neurons, hiddenlayer_neurons))
bh = np.ones((1, hiddenlayer_neurons))
wout = np.ones((hiddenlayer_neurons, output_neurons))
bout = np.ones((1, output_neurons))


output = main_phylanx(X, output_y, wh, bh, wout, bout, lr, num_iter)

print("output = ", output)
