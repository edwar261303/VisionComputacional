#Alumno: Aucapiña Suvizarreta Edwar
#Codigo: 113572
# 3.

import cv2
import numpy as np
#--leer imagen
image = np.array([
    [0,1,2],
    [3,4,5],
    [6,7,8]
])
#--obtener dimensiones
ancho = image.shape[1] #columnas
alto = image.shape[0] # filas
print(alto,ancho)
#--escalando la imagen 
imageOut = cv2.resize(image,(2*ancho,2*alto), interpolation=cv2.INTER_NEAREST)
#--mostrar las imágenes
print("Matriz original: ")
print(image)
print("Matriz Escalada: ")
print(imageOut)