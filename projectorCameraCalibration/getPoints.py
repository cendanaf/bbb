import cv2
import numpy as np
import collections
import getLines

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Can not open camera")
    exit()

Brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
Contrast = cap.get(cv2.CAP_PROP_CONTRAST)
Saturation = cap.get(cv2.CAP_PROP_SATURATION)
Gain = cap.get(cv2.CAP_PROP_GAIN)
Hue = cap.get(cv2.CAP_PROP_HUE)
Exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
print('[Brightness, Contrast, Saturation, Gain, Hue] =\
      [{}, {}, {}, {}, {}, {}]'.format(
            Brightness, Contrast, Saturation, Gain, Hue, Exposure))

def ProcessImage(image):
    """
    Processes image
    (so I don't have to keep doing it for every corner detection algorithm)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)[1]

    kernel = np.ones((15,1), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((17,3), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    return morph
    
def HarrisCorners(image):
    """
    Finds corner pixels using Harris corner detection
    """
    img = ProcessImage(image)

    corners = cv2.cornerHarris(img, 2, 5, 0.04)
    corners = cv2.dilate(corners, None)
    
    points = [ corners > 0.01 * corners.max() ]
    rows, cols = np.where(points[ points == True ])
    return rows, cols

def SIFTfeatures(image):
    img = ProcessImage(image)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def ShiTomasiCorners(image):
    """
    Finds corner pixels using N strongest corners
    in the image by Shi-Tomasi corner detection
    then Harris corner detection (if flag is specified)
    """
    img = ProcessImage(image)

    gf2t = cv2.goodFeaturesToTrack(img, 144, 0.01, 1,
                                   useHarrisDetector=True, k=0.04)
    gf2t = gf2t.astype(int)
    
    points = []
    for i in range(len(gf2t)):
        x, y = gf2t[i][0]
        points.append((x, y))
    return points

def ClusterPointsByKmeans(points, k=175, **kwargs):
    """
    Clusters points into one center point based on location

    We're expecting a 16x9 checkerboard pattern
    (thus having 144 corners at most)
    """
    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 1, 0.5))
    flags = kwargs.get('flags', cv2.KMEANS_PP_CENTERS)
    attempts = kwargs.get('attempts', 10)
    # Apply KMeans
    compactness,labels,centers = cv2.kmeans(z,k,None,criteria,
                                            10,flags)
    clusters = centers.astype(int)
    return clusters

def GetFourCornerPoints(points):
    """
    Finds quadrilateral corner points from a set of points
    """
    X = []
    Y = []
    for i in range(len(points)):
        y, x = points[i]
        X.append(x)
        Y.append(y)

    cornerIndex = [ X.index(max(X)),
                    Y.index(max(Y)),
                    X.index(min(X)),
                    Y.index(min(Y)) ]
    corners = []
    for i in cornerIndex:
        y, x = points[i]
        corners.append((x, y))

    return corners

    

if __name__ == '__main__':
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
            print('[Brightness, Contrast, Saturation, Gain, Hue, Exposure] = \
                  [{}, {}, {}, {}, {}]'.format(Brightness, Contrast, \
                    Saturation, Gain, Hue, Exposure))
        
        ret, frame = cap.read()
        if not ret:
            print("Can not receive frame (frame end?). Exiting...")
            break

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('t'):
            filename = 'checkerboardPattern.png'
            cv2.imwrite(filename, frame)
            print("Image saved to: {}".format(filename))
        elif key == ord('b'):
            if(Brightness >= 255):
                b = 0
            else:
                b = Brightness + 1
            cap.set(cv2.CAP_PROP_BRIGHTNESS, b)
        elif key == ord('c'):
            if(Contrast >= 255):
               c = 10
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
                g = Saturation + 1
            cap.set(cv2.CAP_PROP_GAIN, g)
        elif key == ord('h'):
            if(Hue >= 255):
               h = 0
            else:
                h = Hue + 1
            cap.set(cv2.CAP_PROP_HUE, h)
        elif key == ord('e'):
            if(Exposure >= 255):
               e = 0
            else:
                e = Exposure + 1
            cap.set(cv2.CAP_PROP_EXPOSURE, e)

    cap.release()
    cv2.destroyAllWindows()
    



    """
    img = cv2.imread('C:\\Users\\fcend\\git\\bbb\\projectorCameraCalibration\\checkerboardPattern.png')

    rows, cols = HarrisCorners(img)
    img[rows, cols] = [255, 255, 0]
    #p = np.array([ [cols[i], rows[i]] for i in range(len(rows)) ])
    p = np.array( list(zip(rows, cols)) )
    pointCorners1 = GetFourCornerPoints(p)
    #for corner in pointCorners1:
    #    y, x = corner
    #    cv2.circle(img, (x, y), radius=1, color=(0,0,255), thickness=-1)

    pointCorners2 = ShiTomasiCorners(img)
    #for corner in pointCorners2:
    #    cv2.circle(img, corner, radius=1, color=(255,255,0), thickness=-1)

    #keypoints, descriptors = SIFTfeatures(img)
    #siftImage = cv2.drawKeypoints(img, keypoints, img)
    
    
    z = np.array([[rows[i], cols[i]] for i in range(len(rows)) ])
    z = np.float32(z)
    clusters = ClusterPointsByKmeans(z)
    #for cluster in clusters:
    #    y, x = cluster
    #    cv2.circle(img, (x, y), radius=1, color=(255,0,0), thickness=-1)

    clusterCorners = GetFourCornerPoints(clusters)
    #for corner in clusterCorners:
    #    cv2.circle(img, corner, radius=2, color=(0,255,0), thickness=-1)
    
    cv2.imshow('Frame', img)
    """
