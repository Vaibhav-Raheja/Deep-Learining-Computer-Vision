"""Perceptron model."""

import numpy as np
import matplotlib.pyplot as plt



class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.losses = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        samples, features = X_train.shape
        self.w = np.zeros((self.n_class, features + 1))
        X_train = np.insert(X_train, 0, 1, axis=1)  # Insert bias term in X_train

        for epoch in range(self.epochs):
            loss = 0
            for idx, x_i in enumerate(X_train):
                real_class = y_train[idx]
                if self.n_class == 2:  # Binary classification
                    y_true = 1 if real_class == 1 else -1
                    linear_b = np.dot(self.w[0], x_i)
                    if y_true * linear_b <= 0:
                        loss += 1
                        self.w[0] += self.lr * y_true * x_i
                else:  # Multi-class classification
                    linear_m = np.dot(self.w, x_i)
                    predicted_class = np.argmax(linear_m)
                    if predicted_class != real_class:
                        loss += 1
                        self.w[real_class] += self.lr * x_i
                        self.w[predicted_class] -= self.lr * x_i
            self.losses.append(loss)
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss}, Lr: {self.lr}')
            self.lr *= 0.1 # Decay


    def plot_loss(self):
        plt.plot(range(1, self.epochs + 1), self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Number of Misclassifications')
        plt.title('Perceptron - Loss vs. Epoch')
        plt.show()


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test = np.insert(X_test, 0, 1, axis=1)  # Insert bias term
        if self.n_class == 2:  # Binary classification
            predictions = np.dot(X_test, self.w[0])
            return np.where(predictions >= 0, 1, 0)
        else:  # Multi-class classification
            predictions = np.dot(X_test, self.w.T)
            return np.argmax(predictions, axis=1)
        
