import cv2
import numpy as np
import collections
import getLines
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def Distance(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    x = (x2 - x1)**2
    y = (y2 - y1)**2
    
    d = np.sqrt(x + y)
    return d

def CreateClusters(nClusters, points):
    kmeans = KMeans(n_clusters=nClusters, n_init='auto', random_state=0).fit(points)
    labels = np.array([[label] for label in kmeans.labels_])
    clusters = np.append(points,labels,axis=1)
    return clusters

def ValidateSolution(maxDist, clusters):
    _, __, nClust = clusters.max(axis=0)
    nClust = int(nClust)
    for i in range(nClust):
        twodCluster=clusters[clusters[:,2] == i][:,np.array([True, True, False])]
        if not ValidateCluster(maxDist, twodCluster):
            return False
        else:
            continue
    return True

def ValidateCluster(maxDist, cluster):
    distances = cdist(cluster,cluster, lambda ori,des: int(round(Distance(ori,des))))
    print(distances)
    print(30*'-')
    for item in distances.flatten():
        if item > maxDist:
            return False
    return True

if __name__ == '__main__':
    img = cv2.imread('C:\\Users\\fcend\\Documents\\checkerboardPattern.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    minLineLength = 100
    maxLineGap = 10
    rho = 1
    theta = np.pi/180
    threshold = 100

    lines = cv2.HoughLines(edges, rho, np.pi/180, threshold)
    segmented = getLines.SegmentByAngleKmeans(lines)

    for line in segmented[0]:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

    for line in segmented[1]:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
    intersections = getLines.SegmentedIntersections(segmented)
    for intersection in intersections:
        cv2.circle(img, (intersection[0], intersection[1]), radius=2, color=(0, 0, 255), thickness=-1)

    #cv2.imshow('Frame', img)

    for i in range(2, len(intersections)):
        print(i)
        print(ValidateSolution(20, CreateClusters(i, intersections)))

