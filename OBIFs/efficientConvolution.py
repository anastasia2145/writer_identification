import cv2 

def efficientConvolution(I,kx,ky):
    J = cv2.filter2D(I, -1, kx)
    J = cv2.filter2D(I, -1, ky)
    return J