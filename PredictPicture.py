import cv2
from PredictImage import *
checkpoint_path = 'weights.38-7.34.hdf5'
model = modelInit(checkpoint_path)

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        #cv2.imshow('raspicam', img)
        y =  predictImg(model, img)
        print(y)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=True)

if __name__== '__main__':
    main()
