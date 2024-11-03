import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a synthetic multi-class classification dataset
X, y = make_classification(n_samples=1000, n_features=4, n_informative=3, n_redundant=0, 
                           n_classes=3, random_state=42)  # 3 classes, 4 features

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling to normalize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert target labels to one-hot encoding for multi-class classification required by categorical_crossentropy
y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_test = keras.utils.to_categorical(y_test, num_classes=3)

# Build a neural network model with Keras for multi-class classification
model = keras.Sequential([
    layers.Input(shape=(4,)),            # Input layer
    layers.Dense(10, activation='relu'), # Hidden layer with 10 neurons and ReLU activation
    layers.Dense(3, activation='softmax') # Output layer with 3 neurons (one for each class) using softmax activation (probabilities that sums to 1)
])

# Compile the model with categorical cross-entropy loss and Adam optimizer
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',  # suitable loss function for multi-class classification
              metrics=['accuracy'])            # accuracy as the evaluation metric

# Fit the model on training data
model.fit(X_train, y_train, 
          epochs=50, 
          batch_size=32, 
          validation_data=(X_test, y_test))

# Evaluate the model on test data
evaluation = model.evaluate(X_test, y_test)
print(f"Test Loss (Categorical Cross-Entropy): {evaluation[0]}, Test Accuracy: {evaluation[1]}")

# Predict on test data and calculate additional metrics
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)   # Convert probabilities to class labels (the index represents the class label) choses the highest index per row of the 2D array
y_true_classes = np.argmax(y_test, axis=1)   # Convert one-hot encoded test labels to class labels

print("Classification Report:\n", classification_report(y_true_classes, y_pred_classes))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")
plt.show()
