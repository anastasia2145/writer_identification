import cv2
import numpy as np

def get_contour(image):
    img = image[:,:,0]
    ret, thresh = cv2.threshold(img,127,255,0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    return contours[0]

def horizontal_hist(image):
    img = image[:,:,0]
    return list(np.sum(img / 255, axis = 0))

def vertical_hist(image):
    img = image[:,:,0]
    return list(np.sum(img / 255, axis = 1))

def upper_profile(image):
    img = image[:,:,0]
    black = np.argwhere(img == 0)
    ind = min(black[:,0])
    return list(img[ind, :] / 255)
    
def lower_profile(image):
    img = image[:,:,0]
    black = np.argwhere(img == 0)
    ind = max(black[:,0])
    return list(img[ind, :] / 255)

def orientation(image):
    cnt = get_contour(image)
    if len(cnt) < 5:
        return 0
    (x,y), (MA,ma), angle = cv2.fitEllipse(cnt)
    return angle
       
def rectangularity(image):
    cnt = get_contour(image)
    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w * h
    return float(area) / rect_area
    
def solidity(image):
    cnt = get_contour(image)
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        hull_area = 1
    return float(area) / hull_area
    
def eccentricity(image):
    img = image[:,:,0]
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    cnt = get_contour(image)
    area = cv2.contourArea(cnt)
    moments = cv2.moments(thresh, 1)
    if area == 0:
        area = 1
    return (((moments['m02'] - moments['m02']) ** 2) + 4 * moments['m11']) / area

def elongation(image):
    img = image[:,:,0]
    cnt = get_contour(image)
    area = cv2.contourArea(cnt)
    kernel = np.ones((3,3), np.uint8)
    iteration_count = 1
    erosion = cv2.erode(img, kernel, iterations = 1)
    while 255 in erosion:
        erosion = cv2.erode(erosion, kernel, iterations = 1)
        iteration_count += 1
    return area / (2 * iteration_count * iteration_count)
    
def perimeter(image):
    cnt = get_contour(image)
    perimeter = cv2.arcLength(cnt, True)
    return perimeter

def get_features(image):
    feature = horizontal_hist(image)
    feature += vertical_hist(image)
    feature += upper_profile(image)
    feature += lower_profile(image)
    feature.append(orientation(image))
    feature.append(rectangularity(image))
    feature.append(solidity(image))
    feature.append(eccentricity(image))
    feature.append(elongation(image))
    feature.append(perimeter(image))
    feature = np.array(feature)
    return feature / np.max(feature)