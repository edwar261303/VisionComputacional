#Tarea Guia 10
#Alumno: Aucapiña Suvizarreta Edwar
#Codigo: 113572
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Ruta de la carptea
ruta="imageP10/"

template = cv2.imread(ruta+"template.png")
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

#titulos de la imagen .png
img_title=["Reescalado","Rotado","Oscurecido"]

img=[]
for k in img_title:
    img.append(cv2.cvtColor(cv2.imread("imageP10/"+k+".png"), cv2.COLOR_BGR2RGB))

plt.imshow(template)
#--dimensiones de la plantilla
template.shape
#--lista con los nombres de diferentes métodos de coincidencia de plantillas.
methods =["cv2.TM_CCOEFF" , "cv2.TM_CCOEFF_NORMED", "cv2.TM_CCORR" ,
          "cv2.TM_CCORR_NORMED" , "cv2.TM_SQDIFF" , "cv2.TM_SQDIFF_NORMED"]

#--obtener los resultados para cada metodo
for i in range(3):
    print("---------"+img_title[i]+"----------")
    print("------------------------")  
    for m in methods:
        
      img_copy = img[i].copy()
      method = eval(m)
      res = cv2.matchTemplate(img_copy,template,method)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

      if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
      else:
        top_left = max_loc
      #print(top_left)    

      height, width, channels = template.shape  
      bottom_right = (top_left[0]+width, top_left[1]+height)

      cv2.rectangle(img_copy, top_left, bottom_right, (0,255,0),6)

      plt.subplot(121)
      plt.imshow(res)
      plt.title("TEMPLATE MATCHING MAP")
      plt.subplot(122)
      plt.imshow(img_copy)
      plt.title("TEMPLATE DETECTION")
      plt.suptitle(m)

      plt.show()


  
'''
--CUANDO SE AJUSTA EL TAMAÑO DE LA IMAGEN
Los mejores  metodos son  templete coeficiente de correlacion (TM_CCOEFF) y (TM_CCOEFF_NORMED)
--CUANDO SE GIRA LA IMAGEN
Los mejores  metodos son TM_CCOEFF_NORMED y TM_CCOEFF_NORMED
--CUANDO SE OSCURECE LA IMAGEN
Los mejores metodos son TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR_NORMED, TM_SQDIFF y 
TM_SQDIFF_NORMED


'''