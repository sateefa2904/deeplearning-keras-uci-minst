import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# function to load the MNIST dataset
def load_mnist():
    # load the mnist dataset from keras datasets
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # normalize the images to be in the range [0,1] (important for better training)
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # reshape images to fit the CNN input format (28x28x1)
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    # make labels into 2D arrays with one column
    train_labels = train_labels.reshape(-1, 1)
    test_labels = test_labels.reshape(-1, 1)

    return train_images, train_labels, test_images, test_labels

# function to create and train the CNN model
def create_and_train_model(training_inputs, training_labels, blocks, filter_size, filter_number, region_size, epochs, cnn_activation):
    # get the input shape from training data
    input_shape = training_inputs.shape[1:]  # should be (28, 28, 1) for MNIST

    # create a sequential model
    model = keras.Sequential()
    
    # add the first convolutional layer with input shape
    model.add(layers.Conv2D(filters=filter_number, kernel_size=(filter_size, filter_size),
                        activation=cnn_activation, input_shape=input_shape, padding="same"))



    model.add(layers.MaxPooling2D(pool_size=(region_size, region_size)))

    # add additional convolutional layers based on 'blocks' count
    for _ in range(blocks - 1):
        model.add(layers.Conv2D(filters=filter_number, kernel_size=(filter_size, filter_size),
                                activation=cnn_activation, padding="same"))
        model.add(layers.MaxPooling2D(pool_size=(region_size, region_size)))

    # flatten before feeding into fully connected output layer
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))  # 10 output classes for digits 0-9

    # compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(training_inputs, training_labels, epochs=epochs, verbose=1, validation_split=0.1)

    model.summary()

    return model

# load dataset
(training_inputs, training_labels, test_inputs, test_labels) = load_mnist()

# list of parameter variations to test
experiments = [
    (2, 3, 32, 2, 20, 'relu'),
    (3, 3, 64, 2, 20, 'relu'),
    (4, 5, 64, 2, 25, 'relu'),
    (3, 3, 128, 2, 30, 'relu'),
    (2, 3, 32, 2, 20, 'tanh'),
    (3, 5, 64, 2, 25, 'relu'),
]

best_accuracy = 0
best_params = None

results = []  # Store accuracy and training time results

for blocks, filter_size, filter_number, region_size, epochs, cnn_activation in experiments:
    print(f"\nTesting configuration: blocks={blocks}, filter_size={filter_size}, "
          f"filter_number={filter_number}, epochs={epochs}, activation={cnn_activation}")
    
    # Start timer before training
    start_time = time.time()
    
    # Train the model
    model = create_and_train_model(training_inputs, training_labels, blocks, filter_size, filter_number, region_size, epochs, cnn_activation)
    
    # Evaluate model on test set
    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=0)
    
    # Stop timer after evaluation
    end_time = time.time()
    
    # Calculate accuracy and training time
    accuracy = test_acc * 100
    training_time = end_time - start_time
    
    print(f"Test accuracy: {accuracy:.2f}% - Training time: {training_time:.2f} seconds")

    # Store results for reference
    results.append((blocks, filter_size, filter_number, region_size, epochs, cnn_activation, accuracy, training_time))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = (blocks, filter_size, filter_number, region_size, epochs, cnn_activation)

# Print best configuration
print("\nBest Configuration:")
print(f"blocks={best_params[0]}, filter_size={best_params[1]}, filter_number={best_params[2]}, "
      f"region_size={best_params[3]}, epochs={best_params[4]}, activation={best_params[5]}")
print(f"Achieved Accuracy: {best_accuracy:.2f}%")

# Save results to a file (optional, useful for answers.pdf)
with open("experiment_results.txt", "w") as f:
    f.write("Configuration Results:\n")
    for res in results:
        f.write(f"blocks={res[0]}, filter_size={res[1]}, filter_number={res[2]}, "
                f"region_size={res[3]}, epochs={res[4]}, activation={res[5]}, "
                f"accuracy={res[6]:.2f}%, training_time={res[7]:.2f} seconds\n")
    f.write(f"\nBest Configuration:\nblocks={best_params[0]}, filter_size={best_params[1]}, "
            f"filter_number={best_params[2]}, region_size={best_params[3]}, epochs={best_params[4]}, "
            f"activation={best_params[5]}, accuracy={best_accuracy:.2f}%\n")
