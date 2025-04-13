import numpy as np
from tensorflow import keras

def create_and_train_model(training_inputs, training_labels, layers, units_per_layer, epochs, hidden_activations, batch_size=32):
    num_features = training_inputs.shape[1]
    num_classes = int(np.max(training_labels)) + 1

    model = keras.Sequential()

    if layers == 2:
        model.add(keras.layers.Dense(num_classes, activation='softmax', input_shape=(num_features,)))
    else:
        model.add(keras.layers.Dense(units_per_layer[0], activation=hidden_activations[0], input_shape=(num_features,)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.2))
        
        for i in range(1, layers - 2):
            model.add(keras.layers.Dense(units_per_layer[i], activation=hidden_activations[i]))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(0.2))
        
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(training_inputs, training_labels, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def test_model(model, test_inputs, test_labels, ints_to_labels):
    predictions = model.predict(test_inputs)
    n_samples = test_inputs.shape[0]
    accuracies = []
    
    for test_index in range(n_samples):
        probs = predictions[test_index]
        max_prob = np.max(probs)
        tied_indices = np.where(np.isclose(probs, max_prob))[0]
        
        if len(tied_indices) > 1:
            chosen_index = np.random.choice(tied_indices)
            true_label = int(test_labels[test_index, 0])
            if true_label in tied_indices:
                accuracy = 1.0 / len(tied_indices)
            else:
                accuracy = 0.0
        else:
            chosen_index = tied_indices[0]
            true_label = int(test_labels[test_index, 0])
            accuracy = 1.0 if chosen_index == true_label else 0.0
        
        predicted_class = ints_to_labels[chosen_index]
        actual_class = ints_to_labels[true_label]
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' %
              (test_index, predicted_class, actual_class, accuracy))
        accuracies.append(accuracy)
    
    test_accuracy = np.mean(accuracies)
    print('Classification accuracy on test set: %.2f%%' % (test_accuracy * 100))
    return test_accuracy
