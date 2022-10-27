#Alumno: Aucapiña Suvizarreta Edwar
#Codigo: 113572
import cv2
import numpy as np
from skimage.util import random_noise

# cargar las imágenes
img = cv2.imread("images/insta.jpg")
cv2.imshow("insta",img)

noise_img_gaussian = random_noise(img, mode='gaussian', clip=True) 
noise_img_salt_pepper = random_noise(img, mode='s&p', amount=0.005)

noise_img_gaussian = np.array(255*noise_img_gaussian, dtype = 'uint8')
noise_img_salt_pepper = np.array(255*noise_img_salt_pepper, dtype = 'uint8')

 
#mostrar el ruido en la imagen

cv2.imshow('ruido gaussiano',noise_img_gaussian)
cv2.imshow('ruido sal y pimienta',noise_img_salt_pepper)

#kitar ruido de la imagen con ruido gaussiano
#Eliminacion de ruido Gaussiano por medias no locales
G_NLMeansDenois = cv2.fastNlMeansDenoisingColored(noise_img_gaussian,None,12,10,7,21)
#Eliminacion de ruido Gaussiano por filtro promediado
G_blurred = np.hstack([
 cv2.blur(noise_img_gaussian,(3,3)),
 cv2.blur(noise_img_gaussian,(5,5)),
 cv2.blur(noise_img_gaussian,(9,9))])

#Eliminacion de ruido Gaussiano por filtro median
G_MedianB = cv2.medianBlur(noise_img_gaussian, 5)

#Eliminacion de ruido Gaussiano por filtro gaussiano
G_DenoGauss = cv2.GaussianBlur(noise_img_gaussian,(5,5),cv2.BORDER_DEFAULT)

cv2.imshow('Gausiano con NLMeansDenois',G_NLMeansDenois)
cv2.imshow("Gausiano con blurred",G_blurred)
cv2.imshow("Gausiano con MedianB",G_MedianB)
cv2.imshow("Gausiano con DenoGauss",G_DenoGauss)


##################################################
#kitar ruido de la imagen con  ruido sal y pimmienta
#Eliminacion de ruido sal y pimmienta por medias no locales
SP_NLMeansDenois = cv2.fastNlMeansDenoisingColored(noise_img_salt_pepper,None,12,10,7,21)
#Eliminacion de ruido sal y pimmienta por filtro promediado
SP_blurred = np.hstack([
 cv2.blur(noise_img_salt_pepper,(3,3)),
 cv2.blur(noise_img_salt_pepper,(5,5)),
 cv2.blur(noise_img_salt_pepper,(9,9))])
#Eliminacion de ruido sal y pimmienta por filtro median
SP_MedianB = cv2.medianBlur(noise_img_salt_pepper, 5)
#Eliminacion de ruido sal y pimmienta por filtro gaussiano
SP_DenoGauss = cv2.GaussianBlur(noise_img_salt_pepper,(5,5),cv2.BORDER_DEFAULT)

#mostrar imagenes
cv2.imshow('S&P con NLMeansDenois',SP_NLMeansDenois)
cv2.imshow("S&P con blurred",SP_blurred)
cv2.imshow("S&P con MedianB",SP_MedianB)
cv2.imshow("S&P con DenoGauss",SP_DenoGauss)

'''
Para mi el mejor metodo para la eliminacion de ruido en  ambas imagenes  es  el de
eliminacion de ruido por medias no locales, porque en ambos casos se mantiene
una calidad y mayor y se disminuye en gran medida el ruido
'''
#Nota: se disminuyo el tamaño de la imagen para poder mostrarlo completamente 712x412
cv2.waitKey(0)
cv2.destroyAllWindows()