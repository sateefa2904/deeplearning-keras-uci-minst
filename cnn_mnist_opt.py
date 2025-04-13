import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
def load_mnist():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)
    train_labels = train_labels.reshape(-1, 1)
    test_labels = test_labels.reshape(-1, 1)
    return train_images, train_labels, test_images, test_labels

# Create and train the optimized CNN model
def create_and_train_model(training_inputs, training_labels, blocks, filter_size, filter_number, region_size, epochs, batch_size):
    input_shape = training_inputs.shape[1:]
    model = keras.Sequential()

    # Add convolutional and max pooling layers with batch normalization
    for _ in range(blocks):
        model.add(layers.Conv2D(filter_number, (filter_size, filter_size), activation=None, padding="same", input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D((region_size, region_size)))
        model.add(layers.Dropout(0.3))  # Dropout for regularization

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.4))  # More dropout before final layer
    model.add(layers.Dense(10, activation="softmax"))

    # Use AdamW optimizer with learning rate decay
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=10000, decay_rate=0.9)
    optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(training_inputs, training_labels, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    return model

# Load dataset
(training_inputs, training_labels, test_inputs, test_labels) = load_mnist()

# Experiment settings
experiments = [
    (3, 5, 64, 2, 30, 128),
    (4, 3, 128, 2, 30, 256),
]

best_accuracy = 0
best_params = None

for blocks, filter_size, filter_number, region_size, epochs, batch_size in experiments:
    print(f"\nTesting configuration: blocks={blocks}, filter_size={filter_size}, "
          f"filter_number={filter_number}, epochs={epochs}, batch_size={batch_size}")

    model = create_and_train_model(training_inputs, training_labels, blocks, filter_size, filter_number, region_size, epochs, batch_size)

    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=0)
    accuracy = test_acc * 100
    print(f"Test accuracy: {accuracy:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = (blocks, filter_size, filter_number, region_size, epochs, batch_size)

print("\nBest Configuration:")
print(f"blocks={best_params[0]}, filter_size={best_params[1]}, filter_number={best_params[2]}, "
      f"region_size={best_params[3]}, epochs={best_params[4]}, batch_size={best_params[5]}")
print(f"Achieved Accuracy: {best_accuracy:.2f}%")
