import numpy as np


class Utils(object):
    '''Provides utility methods.

    This class provides methods which are of use for the different modules of the project.
    '''

    @staticmethod
    def create_window_input(example_parameters):
        '''This function returns a label and joins the different word tensors into a single one.

        Considering a fixed window size, there will be multiple word tuples (label, voltage tensor) inside the list.

        :param example_parameters: The list that acts as the window. Tuples of (label,voltage tensor)
        :return: a tuple consisting of a label and the joint voltage tensors.
        '''

        min_value = 0
        inp_matrix = None
        inp_label = None
        if example_parameters is None or len(example_parameters) == 0:
            raise Exception("None/empty list when trying to build the input for a certain window")
        for label, tensor in example_parameters:
            if tensor.shape[1] > min_value:
                min_value = tensor.shape[1]
                inp_label = label
            if inp_matrix is None:
                inp_matrix = tensor
            else:
                inp_matrix = np.concatenate((inp_matrix, tensor), axis=1)
        return inp_label, inp_matrix

    @staticmethod
    def transform_tensor_3d(matrix):
        tensor = np.zeros((1, matrix.shape[0], matrix.shape[1]), np.float32)
        for row in range(0, matrix.shape[0]):
            for column in range(0, matrix.shape[1]):
                tensor[0, row, column] = matrix[row, column]

        return tensor
