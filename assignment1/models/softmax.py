"""Softmax model."""

import numpy as np
import matplotlib.pyplot as plt

class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.loss_history = []

    def z_norm(self, arr):
            return (arr - np.mean(arr, axis=0)) / np.std(arr, axis=0)

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me
        samples, features = X_train.shape
        grad = np.zeros((features, self.n_class))
        score = np.dot(X_train, self.w)
        exp_score = np.exp(score - np.max(score, axis=1, keepdims=True))
        p = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        one_hot = np.eye(self.n_class)[y_train]

        if self.n_class == 2:
            grad = np.dot(X_train.T, p - one_hot)
        else:
            y_train = np.where(y_train == 0, -1, y_train)
            y_train = y_train.reshape(-1, 1)
            grad = np.dot(X_train.T, ((y_train == 1) - p))

        grad += 2 * self.reg_const * self.w
        grad /= samples
        return grad

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        Initialize self.w as a matrix with random numbers in the range [0, 1)

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        X_train = self.z_norm(X_train)
        samples, features = X_train.shape
        self.w = np.random.rand(features, self.n_class)
        batches = 128
        if self.n_class == 2:
            y_train = np.where(y_train == 0, -1, y_train)

        for _ in range(self.epochs):
            for i in range(0, samples, batches):
                X_batch = X_train[i:i + batches]
                y_batch = y_train[i:i + batches]
                grad = self.calc_gradient(X_batch, y_batch)
                self.w -= self.lr * grad  # Update weights
            loss = self.calculate_loss(X_train, y_train)
            self.loss_history.append(loss)
    
    def calculate_loss(self, X, y):
        score = np.dot(X, self.w)
        exp_score = np.exp(score - np.max(score, axis=1, keepdims=True))
        probib = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        one_hot = np.eye(self.n_class)[y]
        loss = -np.mean(np.sum(one_hot * np.log(probib + 1e-12), axis=1)) + self.reg_const * np.sum(self.w ** 2)
        return loss


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
        X_test = self.z_norm(X_test)
        score = np.dot(X_test, self.w)
        Y_preds = np.argmax(score, axis=1)
        if self.n_class == 2:
            Y_preds = np.where(Y_preds == -1, 0, Y_preds)
        return Y_preds

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs + 1), self.loss_history, linestyle='-', color='b', label='Training Loss')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    

