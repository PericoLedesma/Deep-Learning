# LIBRARIES AND FILES
import numpy as np
import matplotlib.pyplot as plt

def Creating_grid(X_train):
    X_train = np.transpose(X_train)
    # Set min and max values and give it some padding
    x_min, x_max = X_train[0, :].min() - 1, X_train[0, :].max() + 1
    y_min, y_max = X_train[1, :].min() - 1, X_train[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    return np.c_[xx.ravel(), yy.ravel()]


def Plot_results(data, data_keras, X_train, y_train, X_grid, MAX_ITERATIONS):
    #PLOT COST EVOLUTION
    fig = plt.figure(figsize=(14, 5))
    # setting values to rows and column variables
    rows = 1
    columns = 3

    # PLOT 1: COST
    fig.add_subplot(rows, columns, 1)
    plt.title('Cost function')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.plot(data['epoch'], data['cost'], 'b', label="My model")
    plt.plot(data_keras['cost'], 'r', label="Keras model")
    plt.legend(loc="upper right")
    plt.xlim([0,MAX_ITERATIONS])

    #PLOT 2: DECISION BOUNDARIES MY MODEL
    X = np.transpose(X_train)
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01 # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = data['output_a'][-1]
    Z = Z.reshape(xx.shape)
    Z = np.around(Z)

    fig.add_subplot(rows, columns, 2)
    plt.title('Decision boundaries ')
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y_train, cmap=plt.cm.Spectral)

    #PLOT 3:  ACCURACY
    fig.add_subplot(rows, columns, 3)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(data_keras['accuracy'], 'r', label="Keras model")
    plt.plot(data['accuracy'], 'b', label="My model")
    plt.legend(loc="upper right")
    plt.xlim([0, MAX_ITERATIONS ])

    plt.show()


def Display_Forward(data, layer, input_of_layer):
    print('----------Layer ', layer, '--> compute of outputs.----------')
    print('Weight matrix:')
    print(data['weights'][layer])
    print('Input matrix:')
    print(input_of_layer)
    print('Bias matrix:')
    print(data['bias'][layer])
    print('Neuron values before act.:')
    print(data['output_h'][layer])
    print('Neuron values activated:')
    print(data['output_a'][layer])

def Display_Cost(data, y_train, ):
    print('NN outputs (one per sample)')
    print(data['output_a'][-1])
    print('Labels')
    print(y_train.ravel())


def Display_Training_1(data, layer):
    print('----------------Layer', layer, '----------------')
    print('Weights before')
    print(data['weights'][layer])
    print('Weights gradient')
    print(data['weights_gradients'][layer])
    print('Bias before')
    print(data['bias'][layer])
    print('Bias gradient')
    print(data['bias_gradients'][layer])

def Display_Training_2(data, layer):
    print('Weights after')
    print(data['weights'][layer])
    print('Bias after')
    print(data['bias'][layer])

def Display_weights(data):
    for layer in data['layers']:
        print('Layer ', layer,' weights:')
        print(data['weights'][layer])

def Display_bias(data):
    for layer in data['layers']:
        print('Layer ', layer, ' bias:')
        print(data['bias'][layer])