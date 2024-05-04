"""Logistic regression model."""


import numpy as np



class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this [DONE!]
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.loss = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me [DONE!]
        return 1/(1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Initialize self.w as a matrix with random numbers in the range [0, 1)

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me [DONE!]
        sample_size, features = X_train.shape
        self.w = np.random.rand(features)

        y_train = np.where(y_train == 0, -1, y_train)

        for epochs in range(self.epochs):
            for i in range(sample_size):
                sgd_update_factor  = self.sigmoid(-y_train[i] * np.dot(self.w, X_train[i]))
                self.w += self.lr * sgd_update_factor  * y_train[i] * X_train[i]
            print(f'Epoch {epochs+1}/{self.epochs}, Lr: {self.lr}')
    
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
        # TODO: implement me [DONE!]
        y_pred = self.sigmoid(np.dot(X_test, self.w))
        predicted_classes  = [0 if y <= self.threshold else 1 for y in y_pred]
        return predicted_classes 