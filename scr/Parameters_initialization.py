# Libraries and functions
import numpy as np
from Display_functions import *

def generate_weights_matrixs(data, NEURONS_PER_HIDDEN_LAYER, NUMBER_FEATURES, DISPLAY_INFO):
    # Initialization of the list with the weights. Weights are transpose already: R^(qxp)=R^(neuron x inputs)
    print('--------------------------------------------------------')
    print('*Generating layer´s weights')
    # example of the xavier weight initialization

    # XAVIER WEIGHTS INITIALIZATION
    # np.random.rand((x_dim, y_dim)) * np.sqrt(1 / (NUMBER_SAMPLES))

    for layer in data['layers']:
        # FIRST
        if layer == 0:  # Creating first layer
            weights_matrix = np.random.RandomState(1).randn(NEURONS_PER_HIDDEN_LAYER[layer], NUMBER_FEATURES)* np.sqrt(1 /2)  # Number of inputs are the features
            # weights_matrix = np.zeros((NEURONS_PER_HIDDEN_LAYER[layer], NUMBER_FEATURES))  # Number of inputs are the features
            data['weights'].append(weights_matrix)
            continue
        # REST
        # Inputs of the layer (p) (outputs from the previous one)
        input_shape_p = data['weights'][layer-1].shape[0]

        # Outputs of the layer (q)
        outputs_q = NEURONS_PER_HIDDEN_LAYER[layer]  # Only hidden layers are stored

        # Creating the weight matrix
        weights_matrix = np.random.RandomState(1).rand(outputs_q, input_shape_p) * np.sqrt(1 /NEURONS_PER_HIDDEN_LAYER[layer-1])
        data['weights'].append(weights_matrix)

    # DISPLAY INFO
    if DISPLAY_INFO:
        Display_weights(data)
    print('*Weights generated')
    print('--------------------------------------------------------')


def generate_bias_matrixs(data, NEURONS_PER_HIDDEN_LAYER, DISPLAY_INFO):
    print('--------------------------------------------------------')
    print('*Generating bias')

    for layer in data['layers']:
        bias = np.zeros((NEURONS_PER_HIDDEN_LAYER[layer], 1))
        # bias = np.random.RandomState(1).rand(NEURONS_PER_HIDDEN_LAYER[layer], 1) * 0.1
        data['bias'].append(bias)


    # DISPLAY INFO
    if DISPLAY_INFO:
        Display_bias(data)
    print('*Generated')
    print('--------------------------------------------------------')