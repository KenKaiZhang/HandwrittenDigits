import cv2
import numpy as np

CATEGORIES = {'0':0, '1':1, '2':2, '3':3,
              '4':4, '5':5, '6':6, '7':7,
              '8':8, '9':9, '+':10, '-':11,
              'x':12, '%':13, 'dec':14, '=':15}

img = cv2.imread("Examples\\5+6.png")




cv2.imshow('Output', img)
cv2.waitKey(0)