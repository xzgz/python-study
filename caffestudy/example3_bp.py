import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    x = np.tanh(x)
    return 1.0 - x ** 2

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # self.biases is a column vector
        # self.weights' structure is the same as in the book: http://neuralnetworksanddeeplearning.com/chap2.html
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def update_mini_batch(self, mini_batch, learning_rate = 0.2):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)"."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # delta_nabla_b is dC/db, delta_nabla_w is dC/dw
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learning_rate/len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate/len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward
        activation = x
        activations = [x]   # list to store all the activations, layer by layer
        zs = []             # list to store all the z vectors, layer by layer
        
        # After this loop, activations = [a0, a1, ..., aL], zs = [z1, z2, ..., zL]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # backward pass
        # delta = deltaL .* sigma'(zL)
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        
        # dC/dbL = delta
        # dC/dwL = deltaL * a(L-1)^T
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        '''Note that the variable l in the loop below is used a little
        differently to the notation in Chapter 2 of the book. Here,
        l = 1 means the last layer of neurons, l = 2 is the
        second-last layer, and so on. It's a renumbering of the
        scheme in the book, used here to take advantage of the fact
        that Python can use negative indices in lists.'''
        # z = z(L-l+1), here, l start from 2, end with self.num_layers-1, namely, L-1
        # delta = delta(L-l+1) = w(L-l+2)^T * delta(L-l+2) .* z(L-l+1)
        # nabla_b[L-l+1] = delta(L-l+1)
        # nabla_w[L-l+1] = delta(L-l+1) * a(L-l)^T
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = self.feedforward(test_data)
        return test_results

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

#### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# derivative of the sigmoid function
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

if __name__ == '__main__':

    nn = Network([2, 2, 2])

    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([[1, 1],
                  [1, 0],
                  [0, 1],
                  [0, 0]])
    
    for k in range(40000):
        if k % 10000 == 0:
            print 'epochs:', k
        # Randomly select a sample.
        i = np.random.randint(X.shape[0])
        nn.update_mini_batch(zip([np.atleast_2d(X[i]).T], [np.atleast_2d(y[i]).T]))

    for e in X:
        print(e, nn.evaluate(np.atleast_2d(e).T))




