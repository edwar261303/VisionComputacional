#Alumno: Aucapiña Suvizarreta Edwar
#Codigo: 113572

# importar librerias
import cv2
import matplotlib.pyplot as plt  
import numpy as np
  
# leer la imagen
img = cv2.imread('images/insta2.jpg')
cv2.imshow('original', img)

#Separar los canales
B, G, R = cv2.split (img) # separación de canales


# ecualizar la imagen
equB = cv2.equalizeHist(B)
equG = cv2.equalizeHist(G)  
equR = cv2.equalizeHist(R)
# show image input vs output
img_equ = cv2.merge ([equB, equG, equR]) # Combinación de canales

cv2.imshow('imagen ecualizada', img_equ)

#mostrar histogramas
titles=['Histograma de la imagen original','Histograma de la imagen ecualizada']
images=(img,img_equ)
color = ('b','g','r')
for k in range(2):
    for i,col in enumerate(color):
        histr = cv2.calcHist([images[k]],[i],None,[256],[0,255])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.title(titles[k])
    plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

#Nota: la imagen inta se redujo a 712x412 para poder observarlo  completo