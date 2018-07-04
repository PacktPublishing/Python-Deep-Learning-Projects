import tensorflow as tf
import pickle
from tensorflow.contrib import rnn


class Model:
    x = None
    y = None
    sequence_length = None
    weights = None
    biases = None
    outputs = None

    def __init__(self, name):
        self.name = name

    def build(self, input_number, sequence_length, layers_number, units_number, output_number):
        self.x = tf.placeholder("float", [None, sequence_length, input_number])
        self.y = tf.placeholder("float", [None, output_number])
        self.sequence_length = sequence_length

        self.weights = {
            'out': tf.Variable(tf.random_normal([units_number, output_number]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([output_number]))
        }

        x = tf.transpose(self.x, [1, 0, 2])
        x = tf.reshape(x, [-1, input_number])
        x = tf.split(x, sequence_length, 0)

        lstm_layers = []
        for i in range(0, layers_number):
            lstm_layer = rnn.BasicLSTMCell(units_number)
            lstm_layers.append(lstm_layer)
        deep_lstm = rnn.MultiRNNCell(lstm_layers)
        
        self.outputs, states = rnn.static_rnn(deep_lstm, x, dtype=tf.float32)

        print "Build model with input_number: {}, sequence_length: {}, layers_number: {}, " \
              "units_number: {}, output_number: {}".format(input_number, sequence_length, layers_number,
                                                           units_number, output_number)

        self.save(input_number, sequence_length, layers_number, units_number, output_number)

    def save(self, input_number, sequence_length, layers_number, units_number, output_number):
        config = {
                     "input_number": input_number,
                     "sequence_length": sequence_length,
                     "layers_number": layers_number,
                     "units_number": units_number,
                     "output_number": output_number
        }
        config_file = open(self.get_config_file_path(), 'w')
        pickle.dump(config, config_file)
        config_file.close()

    def restore(self):
        config_file = open(self.get_config_file_path(), 'r')
        config = pickle.load(config_file)
        config_file.close()

        self.build(config["input_number"], config["sequence_length"], config["layers_number"],
                   config["units_number"], config["output_number"])

    def get_classifier(self):
        return tf.matmul(self.outputs[-1], self.weights['out']) + self.biases['out']

    def get_config_file_path(self):
        return "{}/{}.config".format(self.name, self.name)
