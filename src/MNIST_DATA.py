
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import networkx as nx

class MNIST_DATA:

    # MNIST Settings
    MNIST_PATH = '../data/mnist/'   # Directory of MNIST data files
    THRESH = 100                    # Pixel threshold (0-255)
    SIZE = 28                       # Width / height of MNIST images

    def __init__(self):
        '''
        Load MNIST data set and initialize the MNIST data manager object.
        '''
        mndata = MNIST(self.MNIST_PATH)
        self.train_data, self.train_labels = mndata.load_training()
        self.test_data, self.test_labels = mndata.load_testing()

    def plot_digit(self, idx, train = True):
        '''
        Shows the handwritten digit image, with the graph overlayed on top.
        Use the train flag to set whether the digit is taken from the training
        dataset or the testing dataset.
        '''
        A, coords, img = self.get_digit_graph(idx, train)
        G = nx.from_numpy_matrix(A)
        plt.figure(figsize=(6, 6))
        plt.imshow(img, 'gray')
        nx.draw_networkx(G, coords, None, False, 
            node_size = 20, node_color = 'k')
        plt.show()
        return None

    def get_digit_graph(self, idx, train = True):
        '''
        Converts the binary digit image, whose index is idx (0-5999), to a
        simple graph. Returns the adjacency matrix of the graph, as well
        as vertex coordinates for an embedding of the graph,
        and the original grayscale image.
        ''' 

        # Initialize adjacency matrix of complete lattice graph
        d, I = la.toeplitz(np.eye(1, self.SIZE, 1)), np.eye(self.SIZE)
        A = np.kron(d, I) + np.kron(I, d)

        # Keep only vertices which are colored in the image
        B, img = self.load_digit(idx, train)
        A = A[B, :][:, B]

        # Get embedding
        X, Y = np.meshgrid(np.arange(self.SIZE), np.arange(self.SIZE))
        coords = np.transpose(np.vstack((X.flatten(), Y.flatten())))
        coords = coords[B, :]

        return A, coords, img

    def load_digit(self, idx, train = True):
        '''
        Loads the digit image whose index is idx.
        Returns a binary vector containing the flattened, black-and-white
        image, as well as the grayscale image as an array.
        '''
        if train:
            M = np.array(self.train_data[idx])
        else:
            M = np.array(self.test_data[idx])
        B = M >= self.THRESH
        img = np.reshape(M, (self.SIZE, self.SIZE))
        return B, img