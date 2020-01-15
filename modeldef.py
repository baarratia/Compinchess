from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pylab as plt

#!pip install tensorflow-gpu==2.0.0-beta1
import tensorflow as tf

#!pip install tensorflow_hub
import tensorflow_hub as hub

from tensorflow.keras import layers
import numpy as np
import PIL.Image as Image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from tensorflow.keras.preprocessing import image
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import PIL.Image as Image
from os import path

BATCH_SIZE = 32
MODEL_INCEPTION_V3 = {
    "shape": (299, 299),
    "url": "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
    "preprocessor": inception_v3_preprocess_input
}
MODEL_MOBILENET_V2 = {
    "shape": (224, 224),
    "url": "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
    "preprocessor": mobilenet_preprocess_input
}
def create_model(arch=MODEL_MOBILENET_V2):
  image_input = tf.keras.Input(shape=(*arch["shape"],3), name='img')
  nn = hub.KerasLayer(arch["url"],
                      input_shape=(*arch["shape"],3),
                      trainable=True)(image_input)
  nn = layers.Dense(2050, activation='relu')(nn)
  outputs = []
  for i in range(64):
    out = layers.Dense(13, activation='softmax')(nn)
    outputs.append(out)
  model = tf.keras.models.Model(inputs=image_input, outputs=outputs)
  model.compile(optimizer='adam', loss=["categorical_crossentropy"] * 64, loss_weights=[1.0]*64)
  return model

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, labels, batch_size=BATCH_SIZE, shuffle=True, arch=MODEL_MOBILENET_V2):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.shuffle = shuffle
        self.arch = arch
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index, DATA_PATH):
        'Generate one batch of data'
        start = index*self.batch_size
        end = (index+1)*self.batch_size
        idx_labels = self.labels[start:end]
        X = np.zeros((self.batch_size, *self.arch["shape"], 3))
        for i, label in enumerate(idx_labels):
          # make X
          img = image.load_img(DATA_PATH + "/" + label, target_size=self.arch["shape"])
          a = image.img_to_array(img)
          a = self.arch["preprocessor"](a)
          X[i,] = a
        y = []
        for sq in range(64):
          out = np.zeros((self.batch_size,13))
          for i, label in enumerate(idx_labels):
            fen = path.splitext(label)[0]
            rows = self.fill_ones(fen[:-1]).split("-")
            rows.reverse()
            c = rows[sq // 8][sq % 8]
            idx = self.fen_char_to_idx(c)
            out[i,idx] = 1.0
          y.append(out)
        return X, y

    def fen_char_to_idx(self, c):
      s = "KQRBNPkqrbnp1"
      return s.find(c)

    def fill_ones(self, fen):
      for i in range(8,1,-1):
        fen = fen.replace(str(i), "1"*i)
      return fen

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
          random.shuffle(self.labels)
