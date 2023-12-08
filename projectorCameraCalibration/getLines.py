import cv2
import numpy as np
import collections
    

def SegmentAngleByKmeans(lines, k=2, **kwargs):
    """
    Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # k=2 => horizontal and vertical

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 0.5))
    flags = kwargs.get('flags', cv2.KMEANS_PP_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    compactness, labels, centers = cv2.kmeans(pts, k, None, criteria,
                                              attempts, flags)
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = collections.defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def DrawLines(img, lines, color):
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    return img

def LineIntersection(line1, line2):
    """
    Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def SegmentedIntersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for nextGroup in lines[i+1:]:
            for line1 in group:
                for line2 in nextGroup:
                    intersections.append(LineIntersection(line1, line2)) 

    return intersections





if __name__ == '__main__':
    
    img = cv2.imread('C:\\Users\\fcend\\git\\bbb\\projectorCameraCalibration\\checkerboardPattern.png')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    minLineLength = 200
    maxLineGap = 1
    rho = 1
    theta = np.pi/180
    threshold = 107

    lines = cv2.HoughLines(edges, rho, np.pi/180,
                           threshold, np.array([]), 0, 0)
    segmented = SegmentAngleByKmeans(lines)
    horizontals = segmented[0]
    verticals = segmented[1]

    img = DrawLines(img, horizontals, (0, 255, 255))
    img = DrawLines(img, verticals, (255, 255, 0))

    intersections = SegmentedIntersections(segmented)
    for intersection in intersections:
        cv2.circle(img, (intersection[0], intersection[1]),
                   radius=2, color=(255, 0, 0), thickness=-1)
        
    
    """
    # Failed
    a = 15
    b = 8

    ret, corners = cv2.findChessboardCorners(morph, (a,b), None)
    print(ret)
    """
    """
    
    

    

    verticals = segmented[0]
    horizontals = segmented[1]
    
    for line in verticals:
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

    for line in horizontals:
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
    
    
    intersections = SegmentedIntersections(segmented)
    for intersection in intersections:
        cv2.circle(img, (intersection[0], intersection[1]), radius=2, color=(255, 0, 0), thickness=-1)
    
    """
    cv2.imshow('Frame', img)
