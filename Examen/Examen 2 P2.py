import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('images/frase_escrita.png',0)
img2 = np.copy(img)
template = cv.imread('images/letra_b.png',0)
w, h = template.shape[::-1]
# los 6 methodos
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

img2 = np.copy(img)
method = eval('cv.TM_CCOEFF')
# templado
res = cv.matchTemplate(img,template,cv.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
# para los mtedos sqddif y norm considerea el minimos
if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(img,top_left, bottom_right, 255, 2)

cv.imshow('Imagen', img) 

cv.waitKey(0)
cv.destroyAllWindows()