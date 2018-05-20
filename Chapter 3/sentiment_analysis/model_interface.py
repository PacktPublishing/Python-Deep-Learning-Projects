#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn

# Eval Parameters
batch_size = 64

# Configurable params
checkpoint_dir  = "./runs/1508847544/"
embedding = np.load('fasttext_embedding.npy')

eval_train = False

# Misc Parameters
allow_soft_placement = True
log_device_placement = False


def sentiment_engine(x_raw = ''):
    # Map data into vocabulary
    vocab_path = os.path.join(checkpoint_dir, "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    
    
    print('Load pre-trained word vectors')
    
    
    print("\nEvaluating...\n")
    
    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(os.path.join(checkpoint_dir, 'checkpoints'))
    graph = tf.Graph()
    #with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    
    # Get the placeholders from the graph by name
    input_x = tf.get_default_graph().get_operation_by_name("input_x").outputs[0]
    
    dropout_keep_prob = tf.get_default_graph().get_operation_by_name(
        "dropout_keep_prob").outputs[0]
    embedding_placeholder = tf.get_default_graph().get_operation_by_name(
        'embedding/pre_trained').outputs[0]
    
    # Tensors we want to evaluate
    predictions = tf.get_default_graph().get_operation_by_name(
        "output/predictions").outputs[0]



    all_predictions = []
    x_test = np.array(list(vocab_processor.transform([x_raw])))
    # Generate batches for one epoch
    batches = data_helpers.batch_iter(list(x_test), batch_size, 1, shuffle=False)
    
    for x_test_batch in batches:
        batch_predictions = sess.run(predictions, {
            input_x: x_test_batch,
            embedding_placeholder: embedding,
            dropout_keep_prob: 1.0
        })
        all_predictions = np.concatenate(
            [all_predictions, batch_predictions])
        print all_predictions
        return all_predictions

