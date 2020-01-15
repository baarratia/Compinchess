try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

#!pip install pyyaml h5py  # Required to save models in HDF5 format
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
#import tensorflow as tf
#from tensorflow import keras

#print(tf.version.VERSION)
from modeldef import *
from PredictBoard import *

model = create_model()
# Evaluate the model
#loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
#print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

checkpoint_path = '/Users/benja/Documents/Chespi/Software/Chessboard_Detection/weights.38-7.34.hdf5'
DATA_PATH = '/Users/benja/Documents/Chespi/Software/Chessboard_Detection/DataSet'

BATCH_SIZE = 32
# Loads the weights
model.load_weights(checkpoint_path)

model



labels = glob.glob("DataSet/*-.png")
labels = list(map(lambda l: path.basename(l), labels))
print("Number of labels " + str(len(labels)))
labels_train, labels_val = train_test_split(labels)
training_generator = DataGenerator(labels_train)
validation_generator = DataGenerator(labels_train)


test_X, test_y = validation_generator.__getitem__(0, DATA_PATH)
batch_y = model.predict(test_X)
true_fens = y_to_fens(test_y, BATCH_SIZE)
pred_fens = y_to_fens(batch_y, BATCH_SIZE)

index_to_show = 1
file_name = DATA_PATH + "/" + validation_generator.labels[index_to_show]
print("3D Image")
display(Image(filename=file_name, width=400))
BASE_URL = "https://us-central1-spearsx.cloudfunctions.net/chesspic-fen-image/"
print("2D Ground Truth " + true_fens[index_to_show])
display(SVG(url=BASE_URL+true_fens[index_to_show]))
print("2D Prediction " + pred_fens[index_to_show])
display(SVG(url=BASE_URL+pred_fens[index_to_show]))
