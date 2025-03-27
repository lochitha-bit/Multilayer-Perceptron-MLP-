
# Multilayer Perceptron (MLP) for Wine Classification

## Overview
This project implements a Multilayer Perceptron (MLP) for classifying wine samples based on their chemical properties. The Wine dataset, available in the scikit-learn library, contains 178 samples of wine categorized into three classes. The dataset includes 13 numerical features that describe different chemical compositions.

The MLP model is trained using TensorFlow/Keras and evaluates its performance using accuracy and loss metrics.

## Dataset
- The dataset consists of 178 samples and 13 numerical features.
- The target variable has three classes, representing different types of wine.
- The dataset is preprocessed by normalizing feature values and encoding labels.

## Prerequisites
Ensure you have the following dependencies installed:

- Python (>=3.7)
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- TensorFlow / Keras

To install the required packages, run:
bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras

## Installation and Setup
1. Clone the repository:
bash
   git clone https://github.com/lochitha-bit/Multilayer-Perceptron-MLP-.git


2. Navigate to the project directory:

bash
   cd Multilayer_Perceptron_(MLP).ipynb


3. Run the Jupyter Notebook or Python script:
bash
   jupyter notebook


## Implementation Steps
1. Load the Wine dataset from scikit-learn.
2. Preprocess the data by scaling the features.
3. Split the dataset into training and test sets.
4. Build an MLP model with input, hidden, and output layers.
5. Train the model using backpropagation and the Adam optimizer.
6. Evaluate model performance on the test set.
7. Visualize training history and results.

## Example Code


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build MLP model
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')  # Output layer with 3 classes
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


## Model Performance Metrics
- Accuracy on the test set
- Loss function values over training epochs
- Confusion matrix for classification results

## Visualizations
- Training vs. Validation Accuracy Plot
- Training vs. Validation Loss Plot
- Confusion Matrix to analyze classification performance

## Future Improvements
- Experiment with different activation functions such as Tanh or Leaky ReLU.
- Introduce dropout layers to prevent overfitting.
- Tune hyperparameters such as the number of layers, neurons per layer, and learning rate.
- Extend to deep learning models like Convolutional Neural Networks (CNNs) for more complex datasets.

## License
This project is licensed under the MIT License.
