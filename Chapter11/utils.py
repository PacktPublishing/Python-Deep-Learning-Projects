
import tensorflow as tf
from sklearn.utils import shuffle

import numpy as np
import os
import json


def get_train_data(num_examples=3000):
    annotation_zip = tf.keras.utils.get_file('captions.zip', 
                                              cache_subdir=os.path.abspath('.'),
                                              origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                              extract = True)
    annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
    
    name_of_zip = 'train2014.zip'
    if os.path.exists(os.path.abspath('.') + '/' + name_of_zip) :
        image_zip = tf.keras.utils.get_file(name_of_zip, 
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                          extract = True)
        PATH = os.path.dirname(image_zip)+'/train2014/'
    else:
      print ('Skipped')
      PATH = os.path.abspath('.')+'/train2014/'
  
  
    # read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # storing the captions and the image name in vectors
    all_captions = []
    all_img_name_vector = []
    
    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)
        
        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)
    
    # shuffling the captions and image_names together
    # setting a random state
    train_captions, img_name_vector = shuffle(all_captions,
                                              all_img_name_vector,
                                              random_state=1)

    # selecting the first 30000 captions from the shuffled set
    
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]
    return train_captions, img_name_vector



def load_image(image_path):
    img = tf.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path



def get_inception_model():
    image_model = tf.keras.applications.InceptionV3(include_top=False, 
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    return image_features_extract_model



def image_to_feature(img_name_vector,image_features_extract_model):
    # getting the unique images
    encode_train = sorted(set(img_name_vector))
    
    # feel free to change the batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train).map(load_image).batch(16)
    
    for img, path in image_dataset:
      batch_features = image_features_extract_model(img)
      batch_features = tf.reshape(batch_features, 
                                  (batch_features.shape[0], -1, batch_features.shape[3]))
    
      for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())


def text_to_vec(train_captions):
    top_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, 
                                                      oov_token="<unk>", 
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
#    train_seqs = tokenizer.texts_to_sequences(train_captions)
    
    
    tokenizer.word_index = {key:value for key, value in tokenizer.word_index.items() if value <= top_k}
    # putting <unk> token in the word2idx dictionary
    tokenizer.word_index[tokenizer.oov_token] = top_k + 1
    tokenizer.word_index['<pad>'] = 0
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    return  tokenizer,cap_vector
