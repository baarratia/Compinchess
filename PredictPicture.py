import cv2
from PredictImage import *
checkpoint_path = 'model_brown_60.h5'
model = modelInit(checkpoint_path)

#scale_percent = 47

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        #width = int(img.shape[1] * scale_percent / 100)
        #height = int(img.shape[0] * scale_percent / 100)
        #dim = (width, height)
        #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #crop = resized[0:224,38:262]
        cv2.imshow('raspicam', img)
        #print(crop.shape)
        y =  predictImg(model, img)
        print(y)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=True)

if __name__== '__main__':
    main()
