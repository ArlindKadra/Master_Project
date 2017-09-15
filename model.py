import pickle
import numpy as np
from utils import Utils
from converter import Converter
from sklearn import preprocessing


class Input(object):
    '''This is the main class that deals with the data.

    This class will organize, manipulate and feed the data to the network.

    Attributes:
        train: A list which contains the training data.
        validation: A list which contains the validation data.
        test: A list which contains the test data.
        converter: An instance of the converter, it will be used to convert the labels into the index representation.
    '''

    def __init__(self):

        self.train = []
        self.validation = []
        self.test = []
        self.split_data(self.group_same_word())
        self.converter = Converter(self.train)

    def read_data(self):
        with open("trainingData.pickle", "rb") as file:
            return pickle.load(file)

    def group_same_word(self):
        '''
        Group voltage matrices with the same word name under one key in the dictionary.
        :return:
        '''

        data = self.read_data()
        dictionary = dict()
        # pass by every tuple which consists of the word name and the voltage matrix over the channels
        for i in range(0, len(data)):
            label, matrix = data[i]
            label = str.lower(label)
            # When the word is not present in the dictionary, create an empty list with that word as a key.
            if label not in dictionary:
                dictionary[label] = []
            dictionary[label].append(matrix)
        return dictionary

    def split_data(self, data):
        '''Separate into training data, validation data, test data.

        Split the data set into data available for training, validation and test.
        Remove words which have lower occurrences than 5.

        :param data: A dictionary, it contains labels as keys. The values are lists with voltage matrices.
        :return:
        '''

        training_data = dict()
        validation_data = dict()
        test_data = dict()
        # go through each word
        for key in data.keys():
            # get the number of word occurrences
            size = len(data[key])
            if size >= 5:
                word_matrices = data[key]
                test_data_index = int(size/5)
                validation_data_index = 2 * test_data_index
                test_data[key] = word_matrices[0:test_data_index]
                validation_data[key] = word_matrices[test_data_index:validation_data_index]
                training_data[key] = word_matrices[validation_data_index:]

        self.train = self.make_pairs(training_data)
        self.validation = self.make_pairs(validation_data)
        self.test = self.make_pairs(test_data)

    def make_pairs(self, data):
        '''Prepare the data as a tuple (matrix, label).

        :param data: A dictionary, it contains labels as keys. The values are lists with voltage matrices.
        :return: A list with tuples, each tuple contains a normalized voltage matrix and a label.
        '''

        temp = []
        for label in data.keys():
            for matrix in data[label]:
                temp.append((preprocessing.normalize(matrix, norm='l2'), label))
        return temp

    def prepare_data(self, data):
        '''From the data build a list for the examples and a list for the labels.

        Transform each 2D matrix into a 3D tensor with a depth of one.
        Convert each label into an index representation, each index is a unique number.

        :param data: A list with tuples, each tuple contains a normalized voltage matrix and a label.
        :return: A tuple of lists.
        '''

        inputs = []
        labels = []
        for label,matrix in data:
            inputs.append(Utils.transform_tensor_3d(matrix))
            labels.append(self.converter.convert(label))
        return (inputs,labels)

    def get_nr_classes(self):
        return self.converter.get_nr_classes()

    def get_batch(self, phase, window_size=173, batch_size=50):
        '''Get a batch from the data.

        This method will implement the logic of the sliding window.
        The data is shuffled and a sliding window is passed through it to create the final input.
        The window will be an example, it will contain 1 or more words.
        The label of the window will be the word which contributes more to the window.
        For the purpose of data augmentation this is generated each time the network requires a batch.

        :param phase: Indicating whether the network is requesting data for the train/validation/test phase.
        :param window_size: The sliding window size. Default value 173.
        :param batch_size: The batch size. Default value 50.
        :return: Returns the batches one by one until there are no more.
        '''

        input = []
        if (phase == "train"):
            input = self.train
        elif (phase == "validation"):
            input = self.validation
        elif (phase == "test"):
            input = self.test

        window = []
        batch = []
        indices = np.arange(len(input))
        np.random.shuffle(indices)
        length = window_size
        for index in indices:
            matrix, label = input[index]
            tensor_length = (matrix.shape[1])
            i = 0
            while True:
                if i + length > tensor_length:
                    length -= (tensor_length-i)
                    window.append((label, matrix[:, i:tensor_length ]))
                    break
                elif i + length == tensor_length:
                    window.append((label, matrix[:, i:tensor_length]))
                    # append one example
                    batch.append(Utils.create_window_input(window))
                    # clear the list that contains the data for the window
                    window.clear()
                    length = window_size
                    break
                elif i + length < tensor_length:
                    window.append((label, matrix[:, i:i + length]))
                    # append one example
                    batch.append(Utils.create_window_input(window))
                    # clear the list that contains the data for the window
                    window.clear()
                    i += length
                    length = window_size

        examples, labels = self.prepare_data(batch)
        for i in range(0, (len(batch) - batch_size), batch_size):
            yield examples[i:i + batch_size], labels[i:i + batch_size]
