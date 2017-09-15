import theano
import theano.tensor as T
import logging
import lasagne
import time

from model import Input
from plot import Plot

def main():

    logging.basicConfig(filename="network.log", level=logging.INFO)
    logging.info("Started")
    CNN(100)

class CNN:
    '''The main class of the project. This class creates the network architecture.

    Attributes:
        input: An instance of the Input class which contains the data and feeds the network.
        graph: The plot object which will plot the train and val loss.
    '''
    def __init__(self, epochs=10):

        self.input = Input()
        self.graph = Plot(epochs)
        self.create(epochs)

    def create(self, num_epochs=10):

        input_var = T.tensor4('inputs')
        target_var = T.lvector('targets')
        network = self.build(input_var)
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        #l2_penalty = 0.005 * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        #loss += l2_penalty
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01)
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
        train_fn = theano.function([input_var, target_var], [loss, train_acc], updates=updates)
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
        
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_accuracy = 0
            train_batches = 0
            start_time = time.time()
            for inputs, targets in self.input.get_batch("train"):
                err, acc = train_fn(inputs, targets)
                train_err += err
                train_accuracy += acc
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for inputs, targets in self.input.get_batch("validation"):
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we log the results for this epoch: # Finally, launch the training loop.
            logging.info("Epoch %d of %d took %.3f s - Training loss %.6f acc %.3f%% Validation loss %.6f acc %.3f%%", (epoch + 1), num_epochs, (time.time() - start_time), (train_err / train_batches), (train_accuracy / train_batches) * 100, (val_err / val_batches), (val_acc / val_batches) * 100)
            self.graph.add_point((train_err / train_batches), (val_err / val_batches))
        # After training, we compute the test loss and accuracy
        test_err = 0
        test_acc = 0
        test_batches = 0
        for inputs, targets in self.input.get_batch("test"):
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        logging.info("Final results: Test loss %.6f acc %.2f", (test_err / test_batches), (test_acc / test_batches) * 100)
        self.graph.save_plot("plot.png")
        # Dumping network parameters
        #with open("parametersCNNVanilla.pickle" , "w") as parameterFile:
            #pickle.dump(lasagne.layers.get_all_param_values(network), parameterFile)

    def build(self, input_var=None):

        network = lasagne.layers.InputLayer(shape=(50, 1, 64, 173), input_var=input_var)
        network = lasagne.layers.Conv2DLayer(network, num_filters=50, filter_size=(1, 5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        network = lasagne.layers.batch_norm(network)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, 2))
        network = lasagne.layers.Conv2DLayer(network, num_filters=40, filter_size=(1, 5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        network = lasagne.layers.batch_norm(network)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, 2))
        network = lasagne.layers.Conv2DLayer(network, num_filters=20, filter_size=(1, 3), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        network = lasagne.layers.batch_norm(network)
        network = lasagne.layers.DenseLayer(network, num_units=400, nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.DenseLayer(network, num_units= self.input.get_nr_classes(), nonlinearity=lasagne.nonlinearities.softmax)
        return network

if __name__ == "__main__":
    main()
