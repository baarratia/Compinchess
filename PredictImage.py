from __future__ import absolute_import, division, print_function, unicode_literals
import os
import glob
from modeldef import *
from PredictBoard import *


checkpoint_path = 'weights.38-7.34.hdf5'
IMG_PATH = '1B1b2KQ-1q1R1k2-kRRrP1Bq-pPq1Np1r-2bnQqpK-2Rb2KN-4b1B1-1n2qpP1-.png'


def modelInit(checkpoint_path):
    model = create_model()
    model.load_weights(checkpoint_path)
    return model

def predictImg(model, IMG_PATH):
    arch=MODEL_MOBILENET_V2
    img = image.load_img(IMG_PATH, target_size= arch["shape"])
    a = image.img_to_array(img)
    a = arch["preprocessor"](a)
    X = np.zeros((1, *arch["shape"], 3))
    X[0,] = a
    y = model.predict(X)
    pred_fens = y_to_fens(y, 1)[0]
    return pred_fens


#BASE_URL = "https://us-central1-spearsx.cloudfunctions.net/chesspic-fen-image/"
#print("2D Prediction " + pred_fens)
#display(SVG(url=BASE_URL+pred_fens))
