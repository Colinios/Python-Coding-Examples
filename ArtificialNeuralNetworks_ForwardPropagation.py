# Artificial Neural Networks - Forward Propagation #

import numpy as np

weights = np.around(np.random.uniform(size=6), decimals=2)
biases = np.around(np.random.uniform(size=3), decimals=2)
x_1 = 0.5 # input 1
x_2 = 0.85 # input 2
print('x1 is {} and x2 is {}'.format(x_1, x_2))

#Computing the wighted sum of the inputs, 𝑧1,1, at the first node of the hidden layer
z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))

#Computing the weighted sum of the inputs, 𝑧1,2, at the second node of the hidden layer
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_12, decimals=4)))

#Activation functions (Sigmoid)
a_11 = 1.0 / (1.0 + np.exp(-z_11))
a_12 = 1.0 / (1.0 + np.exp(-z_12))

#Activations serve as the inputs to the output layer
z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
a_2 = 1.0 / (1.0 + np.exp(-z_2))
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))



# Initialize a network
n = 2 # number of inputs
num_hidden_layers = 2
m = [2, 2] # number of nodes in each hidden layer
num_nodes_output = 1

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs # number of nodes in the previous layer
    network = {}
    # looping through each layer and randomly initializing the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer] 
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
        num_nodes_previous = num_nodes
    return network


#Generating inputs
from random import seed
import numpy as np

np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)
print('The inputs to the network are {}'.format(inputs))


#Compute node activation
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))


#Forward propagation
def forward_propagate(network, inputs):
    layer_inputs = list(inputs) # starting with the input layer as the input to the first hidden layer
    for layer in network:
        layer_data = network[layer]
        layer_outputs = [] 
        for layer_node in layer_data:
            node_data = layer_data[layer_node]
            # compute the weighted sum and the output of each node at the same time 
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer
    network_predictions = layer_outputs
    return network_predictions


#Application
my_network = initialize_network(7, 3, [3, 8, 1], 2)
inputs = np.around(np.random.uniform(size=7), decimals=2)
predictions = forward_propagate(my_network, inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))


