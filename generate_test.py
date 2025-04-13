import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Define dataset properties
num_train = 300  # More samples for better learning
num_test = 200  
std_dev = 0.5  # Lower standard deviation for even clearer separation
mean_distance = 2.5  # Increase distance to minimize overlapping
train_test_distance = 2.5  # Ensure test set follows similar distribution

# Generate means for the classes
mean_class_1_train = np.array([0, 0])
mean_class_2_train = mean_class_1_train + np.array([mean_distance, mean_distance])

mean_class_1_test = mean_class_1_train + np.array([train_test_distance, train_test_distance])
mean_class_2_test = mean_class_2_train + np.array([train_test_distance, train_test_distance])

# Generate training data
train_class_1 = np.random.normal(loc=mean_class_1_train, scale=std_dev, size=(num_train, 2))
train_class_2 = np.random.normal(loc=mean_class_2_train, scale=std_dev, size=(num_train, 2))

# Generate test data
test_class_1 = np.random.normal(loc=mean_class_1_test, scale=std_dev, size=(num_test, 2))
test_class_2 = np.random.normal(loc=mean_class_2_test, scale=std_dev, size=(num_test, 2))

# Assign labels (0 for class 1, 1 for class 2)
train_data = np.vstack((train_class_1, train_class_2))
train_labels = np.hstack((np.zeros(num_train), np.ones(num_train)))

test_data = np.vstack((test_class_1, test_class_2))
test_labels = np.hstack((np.zeros(num_test), np.ones(num_test)))

# Save to CSV
train_df = pd.DataFrame(train_data, columns=['x1', 'x2'])
train_df['label'] = train_labels
train_df.to_csv("train.csv", index=False)

test_df = pd.DataFrame(test_data, columns=['x1', 'x2'])
test_df['label'] = test_labels
test_df.to_csv("test.csv", index=False)

print("Dataset Generated Successfully!")
print("Training set means:", np.mean(train_class_1, axis=0), np.mean(train_class_2, axis=0))
print("Test set means:", np.mean(test_class_1, axis=0), np.mean(test_class_2, axis=0))
print("Euclidean distance between class means in train:", np.linalg.norm(mean_class_1_train - mean_class_2_train))
print("Euclidean distance between class means in test:", np.linalg.norm(mean_class_1_test - mean_class_2_test))