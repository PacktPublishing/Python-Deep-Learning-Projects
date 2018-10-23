import numpy as np
import codecs

# Class to perform all preprocessing operations

class Preprocessing:
    vocabulary = {}
    binary_vocabulary = {}
    char_lookup = {}
    size = 0
    separator = '->'

# This will take the data file and convert data into one hot encoding and dump the vocab into the file.
    
    def generate(self, input_file_path):
        input_file = codecs.open(input_file_path, 'r', 'utf_8')
        index = 0
        for line in input_file:
            for char in line:
                if char not in self.vocabulary:
                    self.vocabulary[char] = index
                    self.char_lookup[index] = char
                    index += 1
        input_file.close()
        self.set_vocabulary_size()
        self.create_binary_representation()
        
# This method is to load the vocab into the memory
    def retrieve(self, input_file_path):
        input_file = codecs.open(input_file_path, 'r', 'utf_8')
        buffer = ""
        for line in input_file:
            try:
                separator_position = len(buffer) + line.index(self.separator)
                buffer += line
                key = buffer[:separator_position]
                value = buffer[separator_position + len(self.separator):]
                value = np.fromstring(value, sep=',')

                self.binary_vocabulary[key] = value
                self.vocabulary[key] = np.where(value == 1)[0][0]
                self.char_lookup[np.where(value == 1)[0][0]] = key

                buffer = ""
            except ValueError:
                buffer += line
        input_file.close()
        self.set_vocabulary_size()

# Below are some helper functions to perform pre-processing.
    def create_binary_representation(self):
        for key, value in self.vocabulary.iteritems():
            binary = np.zeros(self.size)
            binary[value] = 1
            self.binary_vocabulary[key] = binary

    def set_vocabulary_size(self):
        self.size = len(self.vocabulary)
        print "Vocabulary size: {}".format(self.size)

    def get_serialized_binary_representation(self):
        string = ""
        np.set_printoptions(threshold='nan')
        for key, value in self.binary_vocabulary.iteritems():
            array_as_string = np.array2string(value, separator=',', max_line_width=self.size * self.size)
            string += "{}{}{}\n".format(key.encode('utf-8'), self.separator, array_as_string[1:len(array_as_string) - 1])
        return string
