# Import TensorFlow and enable eager execution
# This code requires TensorFlow version >=1.9

import tensorflow as tf
tf.enable_eager_execution() 
from sklearn.model_selection import train_test_split

import numpy as np
import os
import time

from utils import  get_train_data,get_inception_model,image_to_feature,text_to_vec

from models import CNN_Encoder,RNN_Decoder ,loss_function



print ("Dowloading dataset")
train_captions, img_name_vector = get_train_data()

print ("Dowloading pretrained InceptionV3 weights")
image_features_extract_model = get_inception_model()

print("Transforming images into features")
image_to_feature(img_name_vector,image_features_extract_model)

print ("Transforming text to vectors")
tokenizer,cap_vector = text_to_vec(train_captions)    

print(" Create training and validation sets using 80-20 split")
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector, cap_vector, test_size=0.2, random_state=0)



#########  
#Hyper params
EPOCHS = 4
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index)
features_shape = 2048 # shape of the vector extracted from InceptionV3 is (64, 2048)
attention_features_shape = 64



# loading the numpy files 
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap


print ("Loading data")
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

dataset = dataset.map(lambda item1, item2: tf.py_func(map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=8)

print (" shuffling and batching")
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(1)


print("Loading Encoder")
encoder = CNN_Encoder(embedding_dim)
print("Loading Decoder")
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.train.AdamOptimizer()


checkpoint_dir = './my_model'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
                                 optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder,
                                )

loss_plot = []


for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0
    
    for (batch, (img_tensor, target)) in enumerate(dataset):
        loss = 0
        
        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
        
        with tf.GradientTape() as tape:
            features = encoder(img_tensor)
            
            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)

                loss += loss_function(target[:, i], predictions)
                
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)
        
        total_loss += (loss / int(target.shape[1]))
        
        variables = encoder.variables + decoder.variables
        
        gradients = tape.gradient(loss, variables) 
        
        optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
        
        
        if batch % 100 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, 
                                                          batch, 
                                                          loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / len(cap_vector))
    
    print ('Epoch {} Loss {:.6f}'.format(epoch + 1, 
                                         total_loss/len(cap_vector)))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))



