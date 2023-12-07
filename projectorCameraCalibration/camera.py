import numpy as np
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

Brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
Contrast = cap.get(cv2.CAP_PROP_CONTRAST)
Saturation = cap.get(cv2.CAP_PROP_SATURATION)
Gain = cap.get(cv2.CAP_PROP_GAIN)
Hue = cap.get(cv2.CAP_PROP_HUE)
Exposure = cap.get(cv2.CAP_PROP_EXPOSURE)

print('[Brightness, Contrast, Saturation, Gain, Hue] = [{}, {}, {}, {}, {}]'.format(
            Brightness, Contrast, Saturation, Gain, Hue))

while True:

    # Check if settings have changed
    if( cap.get(cv2.CAP_PROP_BRIGHTNESS) != Brightness or
        cap.get(cv2.CAP_PROP_CONTRAST) != Contrast or
        cap.get(cv2.CAP_PROP_SATURATION) != Saturation or
        cap.get(cv2.CAP_PROP_GAIN) != Gain or
        cap.get(cv2.CAP_PROP_HUE) != Hue or
        cap.get(cv2.CAP_PROP_EXPOSURE) != Exposure):
        Brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        Contrast = cap.get(cv2.CAP_PROP_CONTRAST)
        Saturation = cap.get(cv2.CAP_PROP_SATURATION)
        Gain = cap.get(cv2.CAP_PROP_GAIN)
        Hue = cap.get(cv2.CAP_PROP_HUE)
        Exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        print('[Brightness, Contrast, Saturation, Gain, Hue] = [{}, {}, {}, {}, {}]'.format(
            Brightness, Contrast, Saturation, Gain, Hue))
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('t'):
        cv2.imwrite("checkerboardPattern.png", frame)
        print("Image saved")
    elif key == ord('b'):
        if(Brightness >= 255):
            b = 0
        else:
            b = Brightness + 1
        cap.set(cv2.CAP_PROP_BRIGHTNESS, b)
    elif key == ord('c'):
        if(Contrast >= 255):
            c = 0
        else:
            c = Contrast + 1
        cap.set(cv2.CAP_PROP_CONTRAST, c)
    elif key == ord('s'):
        if(Saturation >= 255):
            s = 0
        else:
            s = Saturation + 1
        cap.set(cv2.CAP_PROP_SATURATION, s)
    elif key == ord('g'):
        if(Gain >= 255):
            g = 0
        else:
            g = Gain + 1
        cap.set(cv2.CAP_PROP_GAIN, g)
    elif key == ord('h'):
        if(Hue >= 255):
            h = 0
        else:
            h = Hue + 1
        cap.set(cv2.CAP_PROP_HUE, h)
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
