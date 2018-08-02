import numpy as np
import time

def main_phython(lr, num_iter):
    # Variable initialization
    # Input array
    a = 7500
    b = 7500
    c = 7500
    X = np.ones((a, b))

    # Output
    y = np.ones((1, a))

    inputlayer_neurons = X.shape[1]  # number of features in data set
    hiddenlayer_neurons = c  # number of hidden layers neurons
    output_neurons = a  # number of neurons at output layer

    # weight and bias initialization
    wh = np.ones((inputlayer_neurons, hiddenlayer_neurons))
    bh = np.ones((1, hiddenlayer_neurons))
    wout = np.ones((hiddenlayer_neurons, output_neurons))
    bout = np.ones((1, output_neurons))

    output = 0
    for i in range(num_iter):
        # print(i)
        hidden_layer_input1 = np.dot(X, wh)
        hidden_layer_input = hidden_layer_input1 + bh
        hiddenlayer_activations = (1 / (1 + np.exp(-hidden_layer_input)))
        output_layer_input1 = np.dot(hiddenlayer_activations, wout)
        output_layer_input = output_layer_input1 + bout
        output = (1 / (1 + np.exp(-output_layer_input)))

        # Backpropagation
        E = y - output
        slope_output_layer = (output * (1 - output))
        slope_hidden_layer = (hiddenlayer_activations * (1 - hiddenlayer_activations))
        d_output = E * slope_output_layer
        Error_at_hidden_layer = np.dot(d_output, np.transpose(wout))  # some problem
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        wout += (np.dot(np.transpose(hiddenlayer_activations), d_output)) * lr
        bout += np.sum(d_output, axis=0, keepdims=True) * lr
        wh += (np.dot(np.transpose(X), d_hiddenlayer)) * lr
        bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr
    return output

time_start = time.time()
output = main_phython(0.1, 800)
time_end = time.time()
print("in time =", time_end - time_start)
