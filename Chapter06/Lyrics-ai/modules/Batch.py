#!/usr/bin/env python

import codecs
from modules.Preprocessing import *

class Batch:
    dataset_full_passes = 0

    def __init__(self, data_file_name, vocabulary_file_path, batch_size, sequence_length):
        self.data_file = codecs.open(data_file_name, 'r', 'utf_8')

        self.vocabulary = Preprocessing()
        self.vocabulary.retrieve(vocabulary_file_path)

        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def get_next_batch(self):
        string_len = self.batch_size * self.sequence_length + self.batch_size
        current_batch = self.data_file.read(string_len)
        batch_vector = []
        label_vector = []

        if len(current_batch) < string_len:
            while len(current_batch) < string_len:
                current_batch += u' '
            self.data_file.seek(0)
            self.dataset_full_passes += 1
            print "Pass {} done".format(self.dataset_full_passes)

        for i in np.arange(0, string_len, self.sequence_length + 1):
            sequence = current_batch[i:i + self.sequence_length]
            label = current_batch[i + self.sequence_length:i + self.sequence_length + 1]
            sequences_vector = []

            for char in sequence:
                sequences_vector.append(self.vocabulary.binary_vocabulary[char])
            batch_vector.append(sequences_vector)
            label_vector.append(self.vocabulary.binary_vocabulary[label])

        return np.asarray(batch_vector), np.asarray(label_vector)

    def clean(self):
        self.data_file.close()
