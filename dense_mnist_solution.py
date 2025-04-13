from tensorflow import keras
import numpy as np

# at the end, code should print overall classification on test set.  (already done in desnse_mnist_base)
def load_mnist():
    # loading MNIST dataset
    (training_inputs, training_labels), (test_inputs, test_labels) = keras.datasets.mnist.load_data()

    # normalizing inputs
    training_inputs = training_inputs.astype('float32') / 255.0
    test_inputs = test_inputs.astype('float32') / 255.0

    # reshaping inputs
    training_inputs = training_inputs.reshape(training_inputs.shape[0], 28 * 28)
    test_inputs = test_inputs.reshape(test_inputs.shape[0], 28 * 28)

    return training_inputs, training_labels, test_inputs, test_labels

def create_and_train_model(training_inputs, training_labels, layers, units_per_layer, epochs, hidden_activations):
    model = keras.Sequential()

    input_shape = training_inputs.shape[1]
    model.add(keras.layers.InputLayer(input_shape=(input_shape,)))

    # hidden layers
    for units, activation in zip(units_per_layer, hidden_activations):
        model.add(keras.layers.Dense(units, activation=activation))

    model.add(keras.layers.Dense(10, activation='softmax'))

    # compiling model
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    # training model
    model.fit(training_inputs, training_labels, epochs=epochs, verbose=1)
    
    return model
