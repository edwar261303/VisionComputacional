import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#--Leer la iamgen
img = cv.imread('images/cambiar-rueda.png', cv.COLOR_BGR2GRAY)
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#--Convertir a escala de grises
grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Detectar bordes
#--Operador Laplaciano

dst = cv.Laplacian(grayImage, cv.CV_16S, ksize = 3)
Laplacian = cv.convertScaleAbs(dst)

#detectar lineas
dst = np.copy(img)
edges = cv.Canny(grayImage,50,500)
lines = cv.HoughLinesP(edges, 2, np.pi/180.0, 50, minLineLength=10, maxLineGap=100)

#--dibujar lineas
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(dst, (x1, y1), (x2, y2), (255,0,0), 1)



#detectar circulos
image = cv.GaussianBlur(img, (5, 5), 0)
cimg = np.copy(image)
thrHold=60
p1 = thrHold
p2 = thrHold * 0.4
circles = cv.HoughCircles(grayImage, cv.HOUGH_GRADIENT, 1, cimg.shape[0]/64, param1=p1, param2=p2, minRadius=25, maxRadius=50)
#--dibujar circulos
if circles is not None:
    cir_len = circles.shape[1] # almacenar la longitud de los círculos encontrados
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Dibuja el círculo exterior
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 0, 255), 2)
        # Dibuja el centro del círculo
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
else:
    cir_len = 0 # no se detectaron círculos
    



cv.imshow('Imagen original', rgb_img) 
cv.imshow('Deteccion de bordes', Laplacian) 
cv.imshow('criculos de imagen', cimg)  
cv.imshow("lineas", dst) 
cv.waitKey(0)
cv.destroyAllWindows()