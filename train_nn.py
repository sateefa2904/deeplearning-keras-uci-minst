import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Extract features and labels
X_train, y_train = train_df[['x1', 'x2']], train_df['label']
X_test, y_test = test_df[['x1', 'x2']], test_df['label']

# Normalize Data (Important for stability)
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Create a more powerful neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),  # More neurons
    BatchNormalization(),  # Normalize activations
    Dense(64, activation='relu'),
    Dropout(0.3),  # Prevent overfitting
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model with Adam optimizer for better performance
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train longer for perfect training accuracy
history = model.fit(X_train, y_train, epochs=300, batch_size=10, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
