#LIBRARIES
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD


def Keras_NN(data_keras, X_train, y_train, TRAINING_STEP, MAX_ITERATIONS_KERAS, NEURONS_PER_LAYER,NUMBER_SAMPLES, BATCH_TRAINING, DISPLAY_INFO):
    print('--------------------------------------------------------')
    print('*KERAS MODEL')

    if BATCH_TRAINING:
        BATCH_SIZE = NUMBER_SAMPLES
    else:
        BATCH_SIZE = 1

    #Definition of the model
    model = Sequential()
    for layer in data_keras['layers']:
        print('Layer', layer, ' Created')
        if layer == 0:
            model.add(Dense(NEURONS_PER_LAYER[layer], input_dim=2, activation='sigmoid')) #In the first layer we have to define the inputs
        else:
            model.add(Dense(NEURONS_PER_LAYER[layer], activation='sigmoid')) #In the last layer we have only one

    #Optimizer configuration. With fixed learning rate
    optimizer = tf.keras.optimizers.SGD(learning_rate=TRAINING_STEP)

    # COMPILE
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # TRAINING
    print('*TRAINING')
    history = model.fit(X_train, y_train, epochs=MAX_ITERATIONS_KERAS, batch_size=BATCH_SIZE, verbose=0)
    print('*Done')

    # Storing data of the training process
    data_keras['cost'] = history.history['loss']
    data_keras['accuracy'] = history.history['accuracy']

    # Evaluate the keras model
    print('+++ Metrics of the keras model.')
    model.evaluate(X_train, y_train, verbose=2)
