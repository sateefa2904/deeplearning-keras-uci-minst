import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from dense_mnist_opt import *  # Using the base solution for now

# Load dataset
(training_inputs, training_labels, test_inputs, test_labels) = load_mnist()

# List of parameter variations to test
experiments = [
    (4, [500, 400, 300, 200], ['relu', 'relu', 'relu', 'relu'], 20, 128),
    (5, [600, 500, 400, 300, 200], ['relu', 'relu', 'relu', 'relu', 'relu'], 30, 128),
    (5, [512, 512, 256, 128, 64], ['relu', 'relu', 'relu', 'relu', 'relu'], 30, 128),
    (6, [600, 500, 400, 300, 200, 100], ['relu', 'relu', 'relu', 'relu', 'relu', 'relu'], 40, 128),
    (5, [600, 500, 400, 300, 200], ['swish', 'swish', 'swish', 'swish', 'swish'], 30, 128),
    (5, [600, 500, 400, 300, 200], ['relu', 'relu', 'relu', 'relu', 'relu'], 30, 256),
]

best_accuracy = 0
best_params = None
results = []  # Store accuracy and training time results

for layers, units_per_layer, hidden_activations, epochs, batch_size in experiments:
    print(f"\nTesting configuration: layers={layers}, units_per_layer={units_per_layer}, "
          f"activations={hidden_activations}, epochs={epochs}, batch_size={batch_size}")
    
    # Start timer before training
    start_time = time.time()
    
    # Train the model
    model = create_and_train_model(training_inputs, training_labels, layers, 
                                   units_per_layer, epochs, hidden_activations)
    
    # Evaluate model on test set
    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=0)
    
    # Stop timer after evaluation
    end_time = time.time()
    
    # Calculate accuracy and training time
    accuracy = test_acc * 100
    training_time = end_time - start_time
    
    print(f"Test accuracy: {accuracy:.2f}% - Training time: {training_time:.2f} seconds")

    # Store results for reference
    results.append((layers, units_per_layer, hidden_activations, epochs, batch_size, accuracy, training_time))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = (layers, units_per_layer, hidden_activations, epochs, batch_size)

# Print best configuration
print("\nBest Configuration:")
print(f"layers={best_params[0]}, units_per_layer={best_params[1]}, activations={best_params[2]}, "
      f"epochs={best_params[3]}, batch_size={best_params[4]}")
print(f"Achieved Accuracy: {best_accuracy:.2f}%")

# Save results to a file (optional, useful for answers.pdf)
with open("experiment_results.txt", "w") as f:
    f.write("Configuration Results:\n")
    for res in results:
        f.write(f"layers={res[0]}, units_per_layer={res[1]}, activations={res[2]}, "
                f"epochs={res[3]}, batch_size={res[4]}, "
                f"accuracy={res[5]:.2f}%, training_time={res[6]:.2f} seconds\n")
    f.write(f"\nBest Configuration:\nlayers={best_params[0]}, units_per_layer={best_params[1]}, "
            f"activations={best_params[2]}, epochs={best_params[3]}, "
            f"batch_size={best_params[4]}, accuracy={best_accuracy:.2f}%\n")
