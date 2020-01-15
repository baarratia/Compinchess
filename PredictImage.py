from __future__ import absolute_import, division, print_function, unicode_literals
import os
import glob
from modeldef import *
from PredictBoard import *


def modelInit(checkpoint_path):
    model = create_model()
    model.load_weights(checkpoint_path)
    return model

def predictImgPATH(model, IMG_PATH):
    arch=MODEL_MOBILENET_V2
    img = image.load_img(IMG_PATH, target_size= arch["shape"])
    a = image.img_to_array(img)
    a = arch["preprocessor"](a)
    X = np.zeros((1, *arch["shape"], 3))
    X[0,] = a
    y = model.predict(X)
    pred_fens = y_to_fens(y, 1)[0]
    return pred_fens

def predictImg(model, img):
    arch=MODEL_MOBILENET_V2
    a = image.img_to_array(img)
    a = arch["preprocessor"](a)
    X = np.zeros((1, *arch["shape"], 3))
    X[0,] = a
    y = model.predict(X)
    pred_fens = y_to_fens(y, 1)[0]
    return pred_fens
