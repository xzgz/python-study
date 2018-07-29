import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x ** 2


class NeuralNetwork:

    def __init__(self, layers, activation = 'tanh'):
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
        # layers = [dim(input), ..., dim(output)]
        # Range of weight values is (-1, 1).
        # Weights of input and hidden layers is random((dim(input) or dim(hidden) + 1, dim(input) or dim(hidden) + 1)).
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1
            self.weights.append(r)
        # Weights of output layer is random((dim(last_hidden) + 1, dim(output)).
        r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate = 0.2, epochs = 200):
        # Add a column of ones to X.
        # This is to add the bias unit to the input layer.
        ones = np.atleast_2d(np.ones(X.shape[0]))
        # X = [1, sample], every row of sample is a sample.
        X = np.concatenate((ones.T, X), axis = 1)

        for k in range(epochs):
            if k % 100 == 0: print 'epochs:', k
            for j in range(X.shape[0]):
                # Randomly select a sample.
#                 i = np.random.randint(X.shape[0])
                a = [X[j]]

                # After this loop, a = [a0^T, a1^T, ..., aL^T]
                for l in range(len(self.weights)):
                    # dot_value is zj^T, j = 1, 2, ..., L(Here, the layer number j start from 0,
                    # and zj is a column vector).
                    # activation is aj^T
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)

                # error = (dC/daL)^T
                error = a[-1] - y[j]
                # This is equal to (dC/daL)^T .* sigma'(zL^T). Finally, deltas = deltaL^T
                deltas = [error * self.activation_prime(a[-1])]

                # After this loop, deltas = [deltaL^T, ..., delta1^T]
                for l in range(len(a) - 2, 0, -1):
                    # This is equal to (w(l+1)^T * delta(l+1))^T .* sigma'(z(l+1)^T), namely deltal^T
                    deltas.append(self.weights[l].dot(deltas[-1]) * self.activation_prime(a[l]))
            
                # After this, deltas = [delta1^T, ..., deltaL^T]
                deltas.reverse()

                # Compute dC/dw and update w
                for l in range(len(self.weights)):
                    # layer = aj^T, j start from 0
                    # delta = delta(j+1)^T
                    # (dC/dw(j+1))^T = aj * delta(j+1)^T, namely, (dC/dw(j+1))^T = layer.T.dot(delta)
                    layer = np.atleast_2d(a[l])
                    delta = np.atleast_2d(deltas[l])
                    self.weights[l] -= learning_rate * layer.T.dot(delta)

    def predict(self, x):
        # a = a0^T
        a = np.concatenate((np.ones(1).T, np.array(x)), axis = 0)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

def f(x):
    return x[0] * x[1]

def threshold(x, td = 0.5):
    if x[0] > td:
        x[0] = 1
    else:
        x[0] = 0
    if x[1] > td:
        x[1] = 1
    else:
        x[1] = 0
    return x


if __name__ == '__main__':
    
    import scipy.io as scio
    import random
    
    data = scio.loadmat('Homework.mat')
    data_source = data['TrainData'].transpose(1, 0)
    label_source = np.array(data['TrainLabel']).transpose(1, 0)
    test_data = np.array(data['TestData']).transpose(1, 0)
    print data_source.shape, label_source.shape, test_data.shape
    test_index = range(data_source.shape[0])
    random.shuffle(test_index)
#     yu = map(f, label_source)
#     print sum(yu)
#     print test_index

    test_source = test_data
    boundary = 400
    train_data = data_source[test_index[:boundary]]
    train_label = label_source[test_index[:boundary]]
    test_data = data_source[test_index[boundary:]]
    test_label = label_source[test_index[boundary:]]

    nn = NeuralNetwork([9, 7, 3, 2])

#     X = np.array([[0, 0],
#                   [0, 1],
#                   [1, 0],
#                   [1, 1]])
# 
# #     y = np.array([0, 1, 1, 0])
#     y = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
# 
    nn.fit(train_data, train_label)
#   
    test_result = []
    for e in test_source:
        test_result.append(nn.predict(e))

    
    test_result = map(threshold, test_result)
    test_result = np.array(test_result).astype('uint8')
#     test = zip(test_data, test_label, test_result)
#     for x, y, z in test:
#         print(x, '-->', y, '-->', z)
#     error = np.linalg.norm(test_label.astype('int8') - test_result)

    test_result = test_result.transpose(1, 0)
    scio.savemat('TestLabel.mat', {'TestLabel':test_result})
    print test_result
    print type(test_result), test_result.shape, test_result.dtype

    result = scio.loadmat('TestLabel.mat')
    result = result['TestLabel']
    print result
    print type(result), result.shape, result.dtype
#     print test_label.astype('int8') - test_result, error
#     for x, y, z, w in test:
#         print(x, '-->', y, '-->', z, '-->', w)
#         print type(nn.predict(e))
#     print test_result, type(test_result)
#         print(e, nn.predict(e))






