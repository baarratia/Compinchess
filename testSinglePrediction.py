from PredictImage import *

checkpoint_path = 'model_brown_60.h5'
IMG_PATH = '1B1b2KQ-1q1R1k2-kRRrP1Bq-pPq1Np1r-2bnQqpK-2Rb2KN-4b1B1-1n2qpP1-.png'

model = modelInit(checkpoint_path)
y =  predictImgPATH(model, IMG_PATH)

print(y)

#show image
#BASE_URL = "https://us-central1-spearsx.cloudfunctions.net/chesspic-fen-image/"
#print("2D Prediction " + pred_fens)
#display(SVG(url=BASE_URL+pred_fens))
