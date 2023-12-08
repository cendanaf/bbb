import cv2
import numpy as np
import collections
import getLines

def GetCorners(image):
    """
    Finds corner pixels using Harris corner detection
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)[1]

    kernel = np.ones((15,1), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((17,3), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    corners = cv2.cornerHarris(morph, 2, 5, 0.04)
    corners = cv2.dilate(corners, None)
    
    points = [ corners > 0.01 * corners.max() ]
    rows, cols = np.where(points[ points == True ])
    return rows, cols

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
    img = cv2.imread('C:\\Users\\fcend\\git\\bbb\\projectorCameraCalibration\\checkerboardPattern.png')
    rows, cols = GetCorners(img)
    img[rows, cols] = [255, 255, 0]
    p = np.array([ [cols[i], rows[i]] for i in range(len(rows)) ])
    pointCorners = GetFourCornerPoints(p)
    for corner in pointCorners:
        y, x = corner
        cv2.circle(img, (x, y), radius=1, color=(0,0,255), thickness=-1)
    
    z = np.array([[rows[i], cols[i]] for i in range(len(rows)) ])
    z = np.float32(z)
    clusters = ClusterPointsByKmeans(z)
    for cluster in clusters:
        y, x = cluster
        cv2.circle(img, (x, y), radius=1, color=(255,0,0), thickness=-1)

    clusterCorners = GetFourCornerPoints(clusters)
    for corner in clusterCorners:
        cv2.circle(img, corner, radius=2, color=(0,255,0), thickness=-1)
    
    """
    topleft = (minx, miny)
    bottomright = (maxx, maxy)
    bottomleft = (minx, maxy)
    topright = (maxx, miny)
    cv2.circle(img, topleft, radius=2, color=(0, 255, 0), thickness=-1)
    cv2.circle(img, topright, radius=2, color=(0, 255, 0), thickness=-1)
    cv2.circle(img, bottomleft, radius=2, color=(0, 255, 0), thickness=-1)
    cv2.circle(img, bottomright, radius=2, color=(0, 255, 0), thickness=-1)
    """
    
    cv2.imshow('Frame', img)
