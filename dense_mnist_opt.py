from tensorflow import keras
import numpy as np

def load_mnist():
    # Load MNIST dataset
    (training_inputs, training_labels), (test_inputs, test_labels) = keras.datasets.mnist.load_data()
    
    # Normalize inputs to [0, 1]
    training_inputs = training_inputs.astype('float32') / 255.0
    test_inputs = test_inputs.astype('float32') / 255.0
    
    # Reshape inputs to work with Dense layers
    training_inputs = training_inputs.reshape(training_inputs.shape[0], 28 * 28)
    test_inputs = test_inputs.reshape(test_inputs.shape[0], 28 * 28)
    
    return training_inputs, training_labels, test_inputs, test_labels

def create_and_train_model(training_inputs, training_labels, layers, units_per_layer, epochs, hidden_activations, batch_size=128):
    model = keras.Sequential()
    input_shape = training_inputs.shape[1]
    model.add(keras.layers.InputLayer(input_shape=(input_shape,)))
    
    # Hidden layers with batch normalization and dropout
    for units, activation in zip(units_per_layer, hidden_activations):
        model.add(keras.layers.Dense(units, activation=activation))
        model.add(keras.layers.BatchNormalization())  # Normalization for stability
        model.add(keras.layers.Dropout(0.2))  # 20% dropout to prevent overfitting
    
    # Output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # Compile model with optimized Adam optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    # Train model with specified batch size
    model.fit(training_inputs, training_labels, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model
