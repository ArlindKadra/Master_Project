import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import pylab
from numpy import arange

class Plot:
    '''This class will plot the Train and Validation loss for the network over the epochs.

    Attributes:
        train_loss: A list with the values over the epochs for the training loss.
        val_loss: A list with the values over the epochs for the validation loss.
        nr_epochs: the number of epochs which will be shown on the x axis.
    '''

    def __init__(self, nr_epochs):

        self.train_loss = []
        self.val_loss = []
        self.nr_epochs = nr_epochs


    def add_point(self, train_point, val_point):

        self.train_loss.append(train_point)
        self.val_loss.append(val_point)

    def save_plot(self, name):

        x = arange(1, self.nr_epochs + 1, 1)
        plt.plot(x, self.train_loss, 'b-', label = 'Train Loss')
        plt.plot(x, self.val_loss, 'r-', label = 'Validation Loss')
        plt.ylabel("Loss")
        pylab.savefig(name, bbox_inches='tight')
