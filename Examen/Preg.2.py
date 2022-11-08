#Alumno: Aucapiña Suvizarreta Edwar
#Codigo: 113572
# 2.

import cv2
import numpy as np
import matplotlib.pyplot as plt
#--leer la imagen
img= cv2.imread('images/fig1.png', 0)
cv2.imshow('Imagen original', img)

#--aplicando la ecualización
histequ = cv2.equalizeHist(img)
#--muestra la imagen ecualizada
cv2.imshow('Imagen ecualizada', histequ)

cv2.waitKey(0)
cv2.destroyAllWindows()