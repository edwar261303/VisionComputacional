#Tarea N° 6
#Alumno: Aucapiña Suvizarreta Edwar
#Codigo: 113572

import cv2
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread("../images/board11.jpg")
img2 = cv2.imread("../images/board22.jpg")

# Convertimos las imagenes a escala de grises
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# umbral global
ret1,th1 = cv2.threshold(img1, 200, 255, cv2.THRESH_BINARY)
ret2,th2 = cv2.threshold(img2, 200, 255, cv2.THRESH_BINARY)

# Se usa la funcion xor entre las 2 matrices
xor1 = cv2.bitwise_xor(th1,th2)
xor2 = cv2.bitwise_xor(img1,img2)

plt.subplot(2,3,1),plt.imshow(xor1,'gray')
plt.title("binarizado_xor")
#plt.subplot(2,3,3),plt.imshow(xor2,'gray')
#plt.title("sin binarizar")

plt.show()