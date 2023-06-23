# Deep Learning Assignment: Lets build a NN from scratch
'''
Structure:
Load data
1. Initialization of the parameters
2. Forward propagation
3. Compute cost
4. Backpropagation
5. Update parameters
'''
# LIBRARIES
import numpy as np

# FILES
from Data_initialization import *
from Parameters_initialization import *
from From_scratch_model import *
from Keras_model import *

# ----------------------------------------------------------------------------
def main():
    # CONTROL PARAMETERS
    VISUALIZE = False  # True to plot the data
    DISPLAY_INFO = False #True to display detail info of the process
    SIMPLE = False # For using simple dataset. False for using the assignment dataset. Check Data_initialization file
    NEURONS_PER_LAYER = [12] # HIDDEN LAYERS FOR BOTH MODELS (DYNAMIC)

    TRAINING_STEP = 0.2 # Training step
    BATCH_TRAINING = False # False to stochastic gradient descent training. Batch need more iterations!
    MAX_ITERATIONS = 10 # Number of iteration of the scratch model
    MAX_ITERATIONS_KERAS = MAX_ITERATIONS # Number of iteration Keras model

    # -----------------------------------------------------------------
    NEURONS_PER_LAYER.append(1) # For the last layer, the output
    NUMBER_LAYERS = len(NEURONS_PER_LAYER) # Hidden layers + output

    # LOAD DATA. X_train[samples,features]
    X_train, y_train, NUMBER_SAMPLES, NUMBER_FEATURES = load_data(SIMPLE, VISUALIZE, DISPLAY_INFO)

    # CREATING LAYERS DIC
    data = creating_data_dict(NUMBER_LAYERS)

    # PARAMETER INITIALIZATION
    generate_weights_matrixs(data, NEURONS_PER_LAYER, NUMBER_FEATURES, DISPLAY_INFO)
    generate_bias_matrixs(data, NEURONS_PER_LAYER, DISPLAY_INFO)

    #Initialicing some parameter for the iteration loop
    epoch = 0
    outputs = np.array([])
    sample = np.array([0])

    while epoch < MAX_ITERATIONS:
        # FORWARD PROPAGATION
        Forward_propagation(data, X_train, DISPLAY_INFO)
        outputs = np.append(outputs, data['output_a'][-1][:,sample]) # Storing values of the epoch for the standard desviation

        #BACK PROPAGATION
        if BATCH_TRAINING == False:
            Back_propagation(data, X_train, y_train, NUMBER_SAMPLES, sample)
        else:
            Back_propagation_batch(data, X_train, y_train, NUMBER_SAMPLES)
            data['standard_desv'].append(0)
            # COST FUNCTION STORING
            Compute_Cost(data, y_train, epoch, SIMPLE, NUMBER_SAMPLES)

            epoch += 1
            print('--------------------------------------------------------')
            print('               EPOCH ', epoch)
            print('--------------------------------------------------------')

        # TRAINING PARAMETERS
        Training_NN(data, TRAINING_STEP, DISPLAY_INFO)

        sample += 1
        if sample == NUMBER_SAMPLES and BATCH_TRAINING == False: #Next epoch

            mean = np.sum(outputs) / NUMBER_SAMPLES
            standard_desviation = np.sqrt(1/NUMBER_SAMPLES * np.sum((outputs - mean)**2))
            data['standard_desv'].append(standard_desviation)

            # COST FUNCTION STORING
            Compute_Cost(data, y_train, epoch, SIMPLE, NUMBER_SAMPLES)

            epoch += 1
            sample = np.array([0])
            outputs = np.array([])
            print('--------------------------------------------------------')
            print('               EPOCH ', epoch)
            print('--------------------------------------------------------')

    # END ITERATION LOOP

    # FINAL COST FUNCTION
    Compute_Cost(data, y_train, 0, SIMPLE, NUMBER_SAMPLES)

    #KERAS MODEL
    data_keras = Keras_creating_dict(data)
    Keras_NN(data_keras, X_train, y_train,TRAINING_STEP, MAX_ITERATIONS_KERAS, NEURONS_PER_LAYER, NUMBER_SAMPLES, BATCH_TRAINING, DISPLAY_INFO)

    #PLOT RESULTS
    X_grid = Creating_grid(X_train)
    Forward_propagation(data, X_grid, DISPLAY_INFO)
    Plot_results(data, data_keras, X_train, y_train, X_grid, MAX_ITERATIONS)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    plt.close('all')
    print('*** Start of the script.')
    main()
    print('----------------------------')
    print('*** End of the script.')
# ----------------------------------------------------------------------------
