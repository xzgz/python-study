import numpy as np
  
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
  
def sigmoid_prime(x):
    return x * (1.0 - x)

def tanh(x):
    return np.tanh(x)
  
def tanh_prime(x):
    return 1.0 - x ** 2
  
  
class NeuralNetwork:
  
    def __init__(self, layers, activation = 'sigmoid'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
  
        # Set weights. It is a list made up of array. Each element of this array is the weights of each layer.
        # The input layer has no weight, so, the size of self.weights is layer_number - 1.
        # self.weights[i] denotes w(i+1)^T
        self.weights = []
        self.biases = []
        # layers = [dim(input), ..., dim(output)]
        # Range of weight values is (-1, 1).
        # Weights of input and hidden layers is random((dim(input) or dim(hidden) + 1, dim(input) or dim(hidden) + 1)).
        for i in range(1, len(layers)):
            r = 2 * np.random.random((layers[i - 1], layers[i])) - 1
            self.weights.append(r)
            r = 2 * np.random.random((1, layers[i])) - 1
            self.biases.append(r)
  
    def fit(self, X, y, learning_rate = 0.2, epochs = 40000):
        # Add a column of ones to X.
        # This is to add the bias unit to the input layer.
#         ones = np.atleast_2d(np.ones(X.shape[0]))
        # X = [1, sample], every row of sample is a sample.
#         X = np.concatenate((ones.T, X), axis = 1)
  
        for k in range(epochs):
            if k % 10000 == 0: print 'epochs:', k
            # Randomly select a sample.
            i = np.random.randint(X.shape[0])
            a = [X[i]]
  
            # After this loop, a = [a0^T, a1^T, ..., aL^T]
            for l in range(len(self.weights)):
                # dot_value is zj^T, j = 1, 2, ..., L(Here, the layer number j start from 0,
                # and zj is a column vector).
                # activation is aj^T
                dot_value = np.dot(a[l], self.weights[l]) + self.biases[l]
                activation = self.activation(dot_value)
                a.append(activation)
  
            # error = (dC/daL)^T
            error = a[-1] - y[i]
            # This is equal to (dC/daL)^T .* sigma'(zL^T). Finally, deltas = deltaL^T
            deltas = [error * self.activation_prime(a[-1])]
  
            # After this loop, deltas = [deltaL^T, ..., delta1^T]
            for l in range(len(a) - 2, 0, -1):
                # This is equal to (w(l+1)^T * delta(l+1))^T .* sigma'(zl^T), namely deltal^T
                deltas.append(self.weights[l].dot(deltas[-1]).T * self.activation_prime(a[l]))
              
            # After this, deltas = [delta1^T, ..., deltaL^T]
            deltas.reverse()
  
            # Compute dC/dw and update w
            for l in range(len(self.weights)):
                # layer = al^T, l start from 0
                # delta = delta(l+1)^T
                # (dC/dw(l+1))^T = al * delta(l+1)^T, namely, (dC/dw(l+1))^T = layer.T.dot(delta)
                layer = np.atleast_2d(a[l])
                delta = np.atleast_2d(deltas[l])
                delta_w = layer.T.dot(delta)
                delta_b = delta
                self.weights[l] -= learning_rate * delta_w
                self.biases[l] -= learning_rate * delta_b
  
    def predict(self, x):
        # a = a0^T
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(x, self.weights[l]) + self.biases[l])
        return a
  
if __name__ == '__main__':
  
    nn = NeuralNetwork([2, 2, 1])
  
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
  
    y = np.array([0, 1, 1, 0])
  
    nn.fit(X, y)
  
    for e in X:
        print(e, nn.predict(e))
   
  


