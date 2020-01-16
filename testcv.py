import cv2

def show_webcam(mirror=False):
    take = True
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('raspicam', img)
        if take is True:
            cv2.imwrite('RNBKQBNR-PPPPPPPP-8-8-8-8-pppppppp-rkbkqbnr-.png', img)
            take = False
            print("saved!")
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    
def main():
    show_webcam()
    
if __name__== '__main__':
    main()
        
