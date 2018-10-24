#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 02:08:10 2018

@author: rahulkumar
"""
import tensorflow as tf
tf.enable_eager_execution()
import pickle ,os
import numpy as np

from utils import  get_inception_model, load_image
from models import CNN_Encoder,RNN_Decoder

with open('token.pkl', 'rb') as infile:
    tokenizer = pickle.load(infile)
    

''' Defining Hyper params'''
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index)
EPOCHS = 4
max_length =40
features_shape = 2048 # shape of the vector extracted from InceptionV3 is (64, 2048)
attention_features_shape = 64




image_features_extract_model = get_inception_model()

index_word = {value:key for key, value in tokenizer.word_index.items()}



encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.train.AdamOptimizer()



checkpoint_dir = './my_model'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
                                 optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder,
                                )

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(index_word[predicted_id])

        if index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

 
    
    
image_url = 'https://tensorflow.org/images/surf.jpg'
image_extension = image_url[-4:]
image_path = tf.keras.utils.get_file('image'+image_extension, 
                                     origin=image_url)

result, attention_plot = evaluate(image_path)
print ('Prediction Caption:', ' '.join(result))
