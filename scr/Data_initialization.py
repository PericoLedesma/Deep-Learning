# Libraries
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
def load_data (SIMPLE, VISUALIZE, DISPLAY_INFO): #Function for generating the data
    print('*Generating data.')
    # Big dataset
    N = 500
    dataset = sklearn.datasets.make_gaussian_quantiles( mean=None,
        cov=0.7,
        n_samples=N,
        n_features=2,
        n_classes=2, shuffle=True, random_state=None)
    X_train = dataset[0]
    y_train = dataset[1]

    #Simple dataset
    if SIMPLE:
        X_train = np.array([(0, 0), (1, 0), (0, 1), (1, 1)])
        y_train = np.array([0, 0, 1, 1])

        #AND
        X_train = np.array([(0, 0), (1, 0), (0, 1), (1, 1)])
        y_train = np.array([0, 0, 0, 1])

        # #AND
        # X_train = np.array([(1, 0), (0, 1), (1, 1)])
        # y_train = np.array([0, 0, 1])

        # # # #XOR
        # X_train = np.array([(0, 0), (1, 0), (0, 1), (1, 1)])
        # y_train = np.array([0, 1, 1, 0])

        # X_train = np.array([(0, 0),(1, 1)])
        # y_train = np.array([0,1])

        # X_train = np.array([(1, 0)])
        # y_train = np.array([1])

    y_train = np.resize(y_train,(y_train.shape[0],1))

    NUMBER_SAMPLES =  X_train.shape[0]
    NUMBER_FEATURES =  X_train.shape[1]

    if DISPLAY_INFO:
        if SIMPLE:
            print('Samples:')
            print(X_train)
            print('Labels:')
            print(y_train)
        print('Number of samples:', NUMBER_SAMPLES)
        print('Number of features:', NUMBER_FEATURES)

    if VISUALIZE:
        data_visualization(X_train, y_train, NUMBER_SAMPLES)

    print('*Data generated.')
    return X_train, y_train, NUMBER_SAMPLES, NUMBER_FEATURES


# ----------------------------------------------------------------------------
def data_visualization(X_train, y_train, NUMBER_SAMPLES):
    print('*Data visualization.')
    dataset = np.concatenate((X_train, y_train), axis=1)
    df = pd.DataFrame(dataset, columns=['x', 'y', 'labels'])
    sns.scatterplot(x="x", y="y", hue='labels', data=df)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()



# ----------------------------------------------------------------------------
def creating_data_dict(NUMBER_LAYERS):
    # Dictionary where to store values
    data = dict()
    data['layers'] = []

    data['weights'] = []
    data['weights_gradients'] = []

    data['bias'] = []
    data['bias_gradients'] = []

    data['output_h'] = []
    data['output_a'] = []

    data['epoch'] = []
    data['cost'] = []
    data['accuracy'] = []

    data['standard_desv'] = []


    # Storing number of layers
    for layer in range(NUMBER_LAYERS):
        data['layers'].append(layer)
    return data

def Keras_creating_dict(data):
    data_keras = dict()
    data_keras['layers'] = data['layers']

    data_keras['accuracy'] = []
    data_keras['history'] = []
    data_keras['cost'] = []
    return data_keras


