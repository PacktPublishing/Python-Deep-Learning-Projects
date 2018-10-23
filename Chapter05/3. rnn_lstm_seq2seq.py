#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np , os
import tensorflow as tf
import collections

# Data Preparation

def build_dataset(words, n_words):
    count = [['GO', 0], ['PAD', 1], ['EOS', 2], ['UNK', 3]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


file_path = './conversation_data/'

with open(file_path+'from.txt', 'r') as fopen:
    text_from = fopen.read().lower().split('\n')
with open(file_path+'to.txt', 'r') as fopen:
    text_to = fopen.read().lower().split('\n')
print('len from: %d, len to: %d'%(len(text_from), len(text_to)))


concat_from = ' '.join(text_from).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(concat_from, vocabulary_size_from)
print('vocab from size: %d'%(vocabulary_size_from))
print('Most common words', count_from[4:10])
print('Sample data', data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])



concat_to = ' '.join(text_to).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, vocabulary_size_to)
print('vocab to size: %d'%(vocabulary_size_to))
print('Most common words', count_to[4:10])
print('Sample data', data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])


GO = dictionary_from['GO']
PAD = dictionary_from['PAD']
EOS = dictionary_from['EOS']
UNK = dictionary_from['UNK']

#Defining seq2seq model
class Chatbot:
    def __init__(self, size_layer, num_layers, embedded_size,
                 from_dict_size, to_dict_size, learning_rate, batch_size):
        
        def cells(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(size_layer,initializer=tf.orthogonal_initializer(),reuse=reuse)
        
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])

        with tf.variable_scope("encoder_embeddings"):        
            
            encoder_embeddings = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
            encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
            main = tf.strided_slice(self.X, [0, 0], [batch_size, -1], [1, 1])
            
        with tf.variable_scope("decoder_embeddings"):        
            decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
            decoder_embeddings = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))
            decoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, decoder_input)
        
        with tf.variable_scope("encoder"):
            rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
            _, last_state = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded,
                                              dtype = tf.float32)
        with tf.variable_scope("decoder"):
            rnn_cells_dec = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
            outputs, _ = tf.nn.dynamic_rnn(rnn_cells_dec, decoder_embedded, 
                                           initial_state = last_state,
                                           dtype = tf.float32)
        with tf.variable_scope("logits"):            
            self.logits = tf.layers.dense(outputs,to_dict_size)
            print(self.logits)
            masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        with tf.variable_scope("cost"):            
            self.cost = tf.contrib.seq2seq.sequence_loss(logits = self.logits,
                                                         targets = self.Y,
                                                         weights = masks)
        with tf.variable_scope("optimizer"):            
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
            
            
            
#Hyperparameters         
            
size_layer = 128
num_layers = 2
embedded_size = 128
learning_rate = 0.001
batch_size = 32
epoch = 1

#Training
tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Chatbot(size_layer, num_layers, embedded_size, vocabulary_size_from + 4, 
                vocabulary_size_to + 4, learning_rate, batch_size)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
checkpoint_dir = os.path.abspath(os.path.join('./', "checkpoints_chatbot"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")

def str_idx(corpus, dic):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            try:
                ints.append(dic[k])
            except Exception as e:
                print(e)
                ints.append(2)
        X.append(ints)
    return X

def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = 50
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(50)
    return padded_seqs, seq_lens

def check_accuracy(logits, Y):
    acc = 0
    for i in range(logits.shape[0]):
        internal_acc = 0
        for k in range(len(Y[i])):
            if Y[i][k] == logits[i][k]:
                internal_acc += 1
        acc += (internal_acc / len(Y[i]))
    return acc / logits.shape[0]

X = str_idx(text_from, dictionary_from)
Y = str_idx(text_to, dictionary_to)

for i in range(epoch):
    total_loss, total_accuracy = 0, 0
    for k in range(0, (len(text_from) // batch_size) * batch_size, batch_size):
        batch_x, seq_x = pad_sentence_batch(X[k: k+batch_size], PAD)
        batch_y, seq_y = pad_sentence_batch(Y[k: k+batch_size], PAD)
        predicted, loss, _ = sess.run([tf.argmax(model.logits,2), model.cost, model.optimizer], 
                                      feed_dict={model.X:batch_x,
                                                model.Y:batch_y,
                                                model.X_seq_len:seq_x,
                                                model.Y_seq_len:seq_y})
        
        total_loss += loss
        total_accuracy += check_accuracy(predicted,batch_y)
#        print 'output:', [rev_dictionary_to[i] for i in predicted[0]]
#        print 'input:', [rev_dictionary_to[i] for i in batch_x[0]]
        
    total_loss /= (len(text_from) // batch_size)
    total_accuracy /= (len(text_from) // batch_size)
    print('epoch: %d, avg loss: %f, avg accuracy: %f'%(i+1, total_loss, total_accuracy))
    path = saver.save(sess, checkpoint_prefix, global_step=i+1)


#Evaluation

def predict(sentence):
    X_in = []
    for word in sentence.split():
        try:
            X_in.append(dictionary_from[word])
        except:
            X_in.append(PAD)
            pass
        
    test, seq_x = pad_sentence_batch([X_in], PAD)
    input_batch = np.zeros([batch_size,seq_x[0]])
    input_batch[0] =test[0] 
        
    log = sess.run(tf.argmax(model.logits,2), 
                                      feed_dict={
                                              model.X:input_batch,
                                              model.X_seq_len:seq_x,
                                              model.Y_seq_len:seq_x
                                              }
                                      )
    
    result=' '.join(rev_dictionary_to[i] for i in log[0])
    return result
    
checkpoint_file = tf.train.latest_checkpoint(os.path.join('./', 'checkpoints_chatbot'))
saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
saver.restore(sess, checkpoint_file)
    
print predict('how are you ?')    



