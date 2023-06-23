# Libraries and functions
import numpy as np
import copy
import math
import random
# random.uniform(0, 1)
from Display_functions import *

def Forward_propagation(data, X_train, DISPLAY_INFO):
    print('--------------------------------------------------------')
    print('*Layer outputs calculations.')

    #We empty the output previous calculated
    data['output_h'] = []
    data['output_a'] = []

    for layer in data['layers']:
        #INPUT OF THE LAYER
        if layer == 0:  # FIRST If it is the first layer, we take the inputs samples
            input_of_layer = np.transpose(X_train)
        else:  # REST. If not, we take the outputs of the previous layer
            input_of_layer = data['output_a'][layer - 1]

        # Neurons values without activation
        data['output_h'].append(np.dot(data['weights'][layer], input_of_layer)) # W.T*x
        data['output_h'][layer] += data['bias'][layer] # + b

        # Activation of the values
        data['output_a'].append(1 / (1 + np.exp(-data['output_h'][layer])))

        # DISPLAY INFO
        if DISPLAY_INFO:
            Display_Forward(data, layer, input_of_layer)

    print('*Outputs calculated.')
    print('--------------------------------------------------------')

def Compute_Cost(data, y_train, epoch, SIMPLE, NUMBER_SAMPLES):
    print('--------------------------------------------------------')
    print('*Cost function.')

    # We store the outputs that are the inputs of the cross entropy loss function
    NN_outputs_a = copy.copy(data['output_a'][-1])
    cost = -np.dot(np.transpose(y_train), np.transpose(np.log(NN_outputs_a))) - np.dot((1 - np.transpose(y_train)), np.transpose(np.log(1-NN_outputs_a)))
    cost = cost / NUMBER_SAMPLES
    # MISCLASSIFICATION
    predicted_misclasification = np.around(data['output_a'][-1]) - y_train.ravel()
    predicted_misclasification = np.sum(np.absolute(predicted_misclasification))
    accuracy = (NUMBER_SAMPLES - predicted_misclasification)/ NUMBER_SAMPLES

    # Storing values
    if epoch != 0:
        data['epoch'].append(epoch)
        data['cost'].append(cost[0])
        data['accuracy'].append(accuracy)

        # DISPLAY INFO
    if SIMPLE:
        Display_Cost(data, y_train)

    print('Cost:', cost)
    print('Number or missclassification:', predicted_misclasification)
    print('Accuracy', accuracy)
    print('Standard desviation:', data['standard_desv'][-1])

    print('*Computed')
    print('--------------------------------------------------------')

def Back_propagation(data, X_train, y_train, NUMBER_SAMPLES, sample_index):
    print('--------------------------------------------------------')
    print('*Back propagation NN')

    #Empty previous values
    data['weights_gradients'] = []
    data['bias_gradients'] = []

    #We have to create first the gradient matrix. Reverse -> cant append
    for layer in data['layers']:
        data['weights_gradients'].append(np.zeros((data['weights'][layer].shape)))
        data['bias_gradients'].append(np.zeros((data['bias'][layer].shape)))

    # Picking a random sample for training
    # sample_index = np.random.randint(low=0, high=NUMBER_SAMPLES, size=(1,))

    # BACK PROPAGATION
    for layer in reversed(data['layers']):
        print('---- Layer ', layer)

        #LAST LAYER COST DERIVATIVE + ACTIVATION FUNCT
        if layer == data['layers'][-1]:
            ACCUMULATIVE_DERIV = data['output_a'][-1][:,sample_index] - np.transpose(y_train[sample_index]) #In columns
            ACCUMULATIVE_DERIV = np.transpose(ACCUMULATIVE_DERIV) #I want it in rows
        else:
            ACCUMULATIVE_DERIV = np.dot(ACCUMULATIVE_DERIV, data['weights'][layer+1]) # Weights trasnpose so rows are neurons inputs. Accumulative in rows like neuros
            ACCUMULATIVE_DERIV = np.transpose(data['output_a'][layer][:,sample_index]*(1-data['output_a'][layer][:,sample_index])) * ACCUMULATIVE_DERIV # We

        # INPUT OF THE LAYER
        if layer == 0:
            LAYER_INPUT = np.transpose(X_train[sample_index,:])
        else:
            LAYER_INPUT = data['output_a'][layer-1][:,sample_index]

        data['weights_gradients'][layer] = np.transpose(np.dot(LAYER_INPUT, ACCUMULATIVE_DERIV))

        data['bias_gradients'][layer] = np.transpose(np.sum(ACCUMULATIVE_DERIV, axis=0)) # In column
        data['bias_gradients'][layer] = data['bias_gradients'][layer].reshape((data['bias_gradients'][layer].shape[0], 1)) # We had a dimension for the columns(n,)

        # SMALL check
        for layer in data['layers']:
            if data['weights'][layer].shape != data['weights_gradients'][layer].shape:
                print('ERROR. Shape gradient not equal to weights shape')
                print('Weights matrix')
                print(data['weights'][layer])
                print('Gradient matrix')
                print(data['weights_gradients'][layer])
                quit()

    print('*Complete')
    print('--------------------------------------------------------')

def Back_propagation_batch(data, X_train, y_train, NUMBER_SAMPLES):
    print('--------------------------------------------------------')
    print('*Back propagation NN')

    #Empty previous values
    data['weights_gradients'] = []
    data['bias_gradients'] = []

    #We have to create first the gradient matrix. Reverse -> cant append
    for layer in data['layers']:
        data['weights_gradients'].append(np.zeros((data['weights'][layer].shape)))
        data['bias_gradients'].append(np.zeros((data['bias'][layer].shape)))

    # BACK PROPAGATION
    for layer in reversed(data['layers']):
        print('---- Layer ', layer)

        #LAST LAYER COST DERIVATIVE + ACTIVATION FUNCT
        if layer == data['layers'][-1]:
            ACCUMULATIVE_DERIV = data['output_a'][-1] - np.transpose(y_train) #In columns
            ACCUMULATIVE_DERIV = np.transpose(ACCUMULATIVE_DERIV) #I want it in rows
        else:
            ACCUMULATIVE_DERIV = np.dot(ACCUMULATIVE_DERIV, data['weights'][layer+1]) # Weights trasnpose so rows are neurons inputs. Accumulative in rows like neuros
            ACCUMULATIVE_DERIV = np.transpose(data['output_a'][layer]*(1-data['output_a'][layer])) * ACCUMULATIVE_DERIV # We

        # INPUT OF THE LAYER
        if layer == 0:
            LAYER_INPUT = np.transpose(X_train)
        else:
            LAYER_INPUT = data['output_a'][layer-1]

        data['weights_gradients'][layer] = np.transpose(np.dot(LAYER_INPUT, ACCUMULATIVE_DERIV)) / NUMBER_SAMPLES

        data['bias_gradients'][layer] = np.transpose(np.sum(ACCUMULATIVE_DERIV, axis=0) / NUMBER_SAMPLES) # In column
        data['bias_gradients'][layer] = data['bias_gradients'][layer].reshape((data['bias_gradients'][layer].shape[0], 1)) # We had a dimension for the columns(n,)

        # SMALL check
        for layer in data['layers']:
            if data['weights'][layer].shape != data['weights_gradients'][layer].shape:
                print('ERROR. Shape gradient not equal to weights shape')
                print('Weights matrix')
                print(data['weights'][layer])
                print('Gradient matrix')
                print(data['weights_gradients'][layer])
                quit()


    print('*Complete')
    print('--------------------------------------------------------')

def Training_NN(data, TRAINING_STEP, DISPLAY_INFO):
    print('--------------------------------------------------------')
    print('*Training NN')

    for layer in data['layers']:
        if DISPLAY_INFO:
            Display_Training_1(data, layer)

        data['weights'][layer] = data['weights'][layer] - TRAINING_STEP * data['weights_gradients'][layer]
        data['bias'][layer] = data['bias'][layer] - TRAINING_STEP * data['bias_gradients'][layer]

        if DISPLAY_INFO:
            Display_Training_2(data, layer)
    print('*Trained')
    print('--------------------------------------------------------')




