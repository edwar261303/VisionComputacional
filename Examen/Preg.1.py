#Alumno: Aucapiña Suvizarreta Edwar
#Codigo: 113572
# 1. Eliminar las líneas más delgadas de la figura (a) y obtener la figura (b) aplicando
#filtrado. Indicar qué tipo de filtro ha utilizado.
#Fig1_a.jpg

import cv2 as cv
#leyendo la imagen cuyo ruido se eliminará usando la función imread()
imageread = cv.imread('images/Fig1_a.jpg')
#usando la función medianBlur() para eliminar el ruido de la imagen dada
imagenormal = cv.medianBlur(imageread, 3)
#mostrando la imagen sin ruido como la salida en la pantalla
cv.imshow('original:', imageread)
cv.imshow('imagen con ruido disminuido',imagenormal)
cv.waitKey(0)
cv.destroyAllWindows()

#Se utilizo un suavizada utilizando la mediana de los pixeles
#medianBlur