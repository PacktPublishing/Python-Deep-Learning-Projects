#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: rahulkumar
"""

from flask import Flask , request, jsonify

import time
from inference import evaluate
import tensorflow as tf

app = Flask(__name__)

 

@app.route("/wowme")
def AutoImageCaption():
    image_url=request.args.get('image')
    print('image_url')
    image_extension = image_url[-4:]
    image_path = tf.keras.utils.get_file(str(int(time.time()))+image_extension, origin=image_url)
    result, attention_plot = evaluate(image_path)
    data = {'Prediction Caption:': ' '.join(result)}
    
    return jsonify(data)

 
if __name__ == "__main__":
    app.run(host = '0.0.0.0',port=8081)
