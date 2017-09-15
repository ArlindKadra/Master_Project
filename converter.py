
class Converter(object):
    '''Convert each label into a unique index.

    Get a list with tuples (example, label), use the labels to build a dictionary which assigns a unique number to each label.
    We use the dictionary to get an index representation for each label.

    Attributes:
        table: A dictionary which maps each label to a unique number
    '''

    def __init__(self, input_pairs):

        self.table = {}
        labels = [label for matrix, label in input_pairs]
        self.create_table(labels)



    def create_table(self, labels):

        classes = list(set(labels))
        for i in range(0, len(classes)):
            self.table[classes[i]] = i

    def convert(self, label):

        return self.table[label]

    def get_nr_classes(self):

        return len(self.table)
