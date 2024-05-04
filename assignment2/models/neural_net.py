"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.t = 0
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        self.m = {}
        self.v = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
            self.m["W" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
            self.v["W" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
            self.m["b" + str(i)] = np.zeros(self.params["b" + str(i)].shape)
            self.v["b" + str(i)] = np.zeros(self.params["b" + str(i)].shape)
            # TODO: You may set parameters for Adam optimizer here
    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        # return X.dot(W) + b
        return np.matmul(X,W)+b
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, de_dz: np.ndarray, reg, N) -> np.ndarray:
        """Gradient of linear layer
            z = WX + b
            returns de_dw, de_db, de_dx
        """
        # TODO: implement me
        de_dw=np.matmul(X.T,de_dz)/N+reg*W/N
        de_dx=np.matmul(de_dz,W.T)/N
        de_db=np.sum(de_dz, keepdims=True, axis=0)/N
        return de_dw, de_db, de_dx


        # dW = np.dot(X.T, de_dz) / N + reg * W / N 
        # db = np.sum(de_dz, axis=0) / N
        # dX = np.dot(de_dz, W.T)
        # return dW, db, dX

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return np.where(X>0,1,0)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        # sig = self.sigmoid(X)
        # return X * (1 - X)
        return np.multiply(X,(1-X))



    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return np.mean(np.square(y-p))


    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return (p-y)



    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        sigmoid_grad = self.sigmoid_grad(p)
        mse_grad = self.mse_grad(y, p)
        return np.multiply(mse_grad,sigmoid_grad)


    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        # self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        # self.cache = {'A0': X}
         # where Z_last is the input to the sigmoid function

        # Handle the first layer separately if needed
        # Assuming the first layer uses ReLU activation, but you can adjust as needed
        # Z1 = self.linear(self.params['W1'], X, self.params['b1'])
        # A1 = self.relu(Z1)
        # self.cache['Z1'] = Z1
        # self.cache['A1'] = A1
        
        # # Iterate through intermediate layers (2 to num_layers-1)
        # for i in range(2, self.num_layers):  # Exclude the last layer for now
        #     Z = self.linear(self.params[f'W{i}'], self.cache[f'A{i-1}'], self.params[f'b{i}'])
        #     A = self.relu(Z)  # Intermediate layers use ReLU
        #     self.cache[f'Z{i}'] = Z
        #     self.cache[f'A{i}'] = A
        
        # # Handle the last layer separately
        # # Assuming the last layer uses sigmoid activation
        # Z_last = self.linear(self.params[f'W{self.num_layers}'], self.cache[f'A{self.num_layers - 1}'], self.params[f'b{self.num_layers}'])
        # A_last = self.sigmoid(Z_last)
        # self.cache[f'Z{self.num_layers}'] = Z_last
        # self.cache[f'A{self.num_layers}'] = A_last
        # A_last = self.sigmoid(Z_last) 
        # A_last = A_last.reshape(-1, self.output_size)

        # return A_last4

        self.cache = {"layer_0": X}
        for i in range(1, self.num_layers):
            W = self.params["W" + str(i)]
            b = self.params["b" + str(i)]
            # print(f"W shape: {W.shape}, X shape: {X.shape}")
            linear_output = self.linear(W, X, b)
            self.cache["layer_h_" + str(i)] = linear_output
            X = self.relu(linear_output)
            self.cache["layer_" + str(i)] = X

        W_last = self.params["W" + str(self.num_layers)]
        b_last = self.params["b" + str(self.num_layers)]
        final_linear_output = self.linear(W_last, X, b_last)
        X = self.sigmoid(final_linear_output)
        self.cache["layer_" + str(self.num_layers)] = X

        return X




    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        # self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.        
        self.gradients = {}

        # Calculate the gradient at the output layer
        de_dz = self.mse_sigmoid_grad(y, self.cache["layer_" + str(self.num_layers)])
        loss = self.mse(y, self.cache["layer_" + str(self.num_layers)])

        for i in range(self.num_layers, 0, -1):
            W = self.params["W" + str(i)]
            b = self.params["b" + str(i)]
            X = self.cache["layer_" + str(i - 1)]

            if i != self.num_layers:  
                de_dz = np.multiply(de_dx, self.relu_grad(self.cache["layer_"+str(i)]))
                de_dw, de_db, de_dx = self.linear_grad(W, X, b, de_dz, 0, 1)
            else:  
              # if i!=1:
              de_dw, de_db, de_dx = self.linear_grad(W, X, b, de_dz, 0, 1)
            
            self.gradients["W" + str(i)] = de_dw
            self.gradients["b" + str(i)] = de_db
            if i != 1:  # No need to calculate de_dx for the first layer as it won't be used
                de_dz = de_dx

        return loss


    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "Adam",
        # opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        print(f"Opt:{opt}, lr:{lr}")

        if opt == "SGD":
            for i in range(1, self.num_layers + 1):
                self.params['W' + str(i)] -= lr * self.gradients['W' + str(i)]
                self.params['b' + str(i)] -= lr * self.gradients['b' + str(i)].reshape(self.params['b' + str(i)].shape)
        elif opt == "Adam":
            self.t += 1
            for i in range(1, self.num_layers + 1):
                self.m['W' + str(i)] = b1 * self.m['W' + str(i)] + (1 - b1) * self.gradients['W' + str(i)]
                self.m['b' + str(i)] = b1 * self.m['b' + str(i)] + (1 - b1) * self.gradients['b' + str(i)]
                self.v['W' + str(i)] = b2 * self.v['W' + str(i)] + (1 - b2) * (self.gradients['W' + str(i)] ** 2)
                self.v['b' + str(i)] = b2 * self.v['b' + str(i)] + (1 - b2) * (self.gradients['b' + str(i)] ** 2)
                mW = self.m['W' + str(i)] / (1 - b1 ** self.t)
                mb = self.m['b' + str(i)] / (1 - b1 ** self.t)
                vW = self.v['W' + str(i)] / (1 - b2 ** self.t)
                v_hat_b = self.v['b' + str(i)] / (1 - b2 ** self.t)
                # print("test")
                self.params['W' + str(i)] -= lr * mW / (np.sqrt(vW) + eps)
                # self.params['b' + str(i)] -= lr * mb / (np.sqrt(v_hat_b) + eps)
                self.params['b' + str(i)] -= lr * mb.reshape(self.params['b' + str(i)].shape) / (np.sqrt(v_hat_b.reshape(self.params['b' + str(i)].shape)) + eps)