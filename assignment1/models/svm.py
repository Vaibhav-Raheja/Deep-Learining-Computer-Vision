"""Support Vector Machine (SVM) model."""

import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.losses = []

    def calc_gradientient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradientient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradientient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        loss = 0
        gradient = np.zeros(self.w.shape)
        samples, features = X_train.shape

        for sample_idx  in range(samples):
            for class_idx in range(self.n_class):
                if class_idx != y_train[sample_idx ]:
                    score_curr_class = np.dot(self.w[class_idx], X_train[sample_idx ])
                    score_true_class = np.dot(self.w[y_train[sample_idx ]], X_train[sample_idx ])
                    margin = score_curr_class - score_true_class + 1
                    if margin > 0:
                        gradient[class_idx] += X_train[sample_idx ]
                        gradient[y_train[sample_idx ]] -= X_train[sample_idx ]
                        loss += margin

        gradient = gradient / samples + self.reg_const * self.w
        loss = np.mean(loss) + 0.5 * self.reg_const * np.sum(self.w ** 2)
        return gradient,loss

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
        samples , features = X_train.shape
        self.w = np.zeros((self.n_class, features))
        X_train = self.norm_data(X_train)
        
        if self.n_class == 2: #Binary
            y_train = np.where(y_train == 0, -1, y_train)
            batches = 256
            decay = 1
        else: #Multi Class
            batches = 256
            decay = 0.5

        for epoch in range(self.epochs):
            for i in range(0, samples, batches):
                X_batch = X_train[i:i + batches]
                y_batch = y_train[i:i + batches]
                gradient_w,loss = self.calc_gradientient(X_batch, y_batch)
                for idx in range(self.n_class):
                    self.w[idx] -= self.lr * gradient_w[idx]
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss}, Lr: {self.lr}')
            self.losses.append(loss) 
            self.lr *= decay


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
        X_test = self.norm_data(X_test)
        Y_preds = np.argmax(np.dot(X_test, self.w.T), axis=1)
        if self.n_class == 2: #Rice
            Y_preds = np.where(Y_preds == -1, 0, Y_preds)
        return Y_preds

    def norm_data(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs + 1), self.losses, linestyle='-', color='b', label='Training Loss')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()