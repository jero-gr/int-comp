# Importar las librerías necesarias
import pandas as pd
import random
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import contingency_matrix
from g04ej01_aux import som_labels

df = pd.read_csv("Guia2/irisbin_trn.csv")
mat_datos = df.to_numpy()
mat_row = len(mat_datos)

# Almacenar los patrones de 1xN en una matriz
pat_aux = mat_datos[:,0:-3]
pat_row = np.shape(pat_aux)[0]
pat_col = np.shape(pat_aux)[1]
pat = np.hstack((pat_aux,np.zeros((pat_row,1))))
# pat = [x_0 x_1 x_2 ... x_n centroide_asociado]

# Definir el color a plotear de acuerdo al tipo de flor que es
clase_set = mat_datos[:,-3]
clase_ver = mat_datos[:,-2]
clase_vir = mat_datos[:,-1]
color_scat = np.where(clase_set==1,'blue',np.where(clase_ver==1,'orange','green'))
# marker_scat = np.where(clase_set==1,'d',np.where(clase_ver==1,'p','v'))

# Columnas a plotear:
col_x = 0
col_y = 2
col_z = 3

# Plotear
fig = plt.figure(1)
ax = plt.axes(projection ="3d")
ax.scatter3D(pat[:,col_x], pat[:,col_y], pat[:,col_z], c=color_scat, marker="o", s=2)

# Definir constantes
epc_max = 100
k_row = 6
ent_col = np.shape(pat)[1]-1
tol = 0.000000001

# Inicializar los pesos de los centroides
'''cent_max = np.amax(pat,axis=0)
cent_min = np.amin(pat,axis=0)
cent_avg = np.mean(pat,axis=0)
cent_abs = cent_max - cent_min
cent = (np.random.random((k_row,1))-0.5) * cent_abs[0] + cent_avg[0]
for i in range(1,ent_col):  # Para cada coordenada
    var = (np.random.random((k_row,1))-0.5) * cent_abs[i] + cent_avg[i]
    cent = np.hstack((cent,var))
'''
# Agarrar k patrones random
cent = pat[np.random.randint(pat.shape[0], size=k_row), :]
cent = cent[:,:-1]
cent_ini = np.copy(cent)

# Plotear los centroides
ax.scatter3D(cent[:,col_x], cent[:,col_y], cent[:,col_z], c="black", marker="x")

dist_abs = np.zeros((k_row,1))
dist_rel = np.ones((k_row,1))

# Recorrer épocas
for epc in range(0,epc_max):

    # Para cada patrón
    for i in range(0,pat_row):
        #dif = np.zeros_like(cent)   # Matriz de diferencias para cada patrón
        dist = np.zeros((k_row)) # Las filas son para cada patrón, las columnas son las distancias a cada centroide

        # Para cada centroide
        for j in range(0,k_row):
            pat_aux = pat[i,:]
            pat_aux = pat_aux[0:ent_col]
            dif = pat_aux - cent[j,:] # Cada elemento es la diferencia con el elemento correspondiente de la matriz de centroides
            dist[j] = np.linalg.norm(dif)

        # Posición del centroide ganador
        argmin_dist = np.argmin(dist)
        pat[i,-1] = argmin_dist

    # Para cada centroide
    cent_prev = cent.copy()    # Guardo el centroide actual antes de cambiarlo para compararlo
    dist_abs_prev = dist_abs.copy()

    for j in range(0,k_row):
        mask = (pat[:,-1] == j)
        pat_j = pat[mask, :]                # Agarro sólo los patrones asociados
        pat_aux = pat_j[:,0:ent_col]
        if len(pat_aux)>0:
            cent[j,:] = pat_aux.mean(axis=0)    # Calculo promedio de cada coordenada y ese es el nuevo centroide

    dif_cent = cent-cent_prev   # diferencia entre centroide actual y centroide previo
    for j in range(0,k_row):
        dist_abs[j] = np.sqrt(np.dot(dif_cent[j,:],dif_cent[j,:]))

    print("Época " + str(epc) + " de " + str(epc_max) + ", dist_abs=" + str(np.sum(dist_abs)))
    if np.sum(dist_abs)<tol:
        break

print(cent)

# Plotear de nuevo
#fig = plt.figure(2)
#ax = plt.axes(projection ="3d")
#ax.scatter3D(pat[:,col_x], pat[:,col_y], pat[:,col_z], c=color_scat, marker="o", s=2)
for i in range(0,len(cent)):
    ax.plot([cent_ini[i,col_x], cent[i,col_x]], [cent_ini[i,col_y], cent[i,col_y]], [cent_ini[i,col_z], cent[i,col_z]], c="black", linewidth=0.8)
ax.scatter3D(cent[:,col_x], cent[:,col_y], cent[:,col_z], c="black", marker="D")

'''
fig2, axs = plt.subplots(2, 3)

longS = 0
anchoS = 1
longP = 2
anchoP = 3

axs[0,0].scatter(pat[:,longS], pat[:,anchoS], c=color_scat, marker="o",s=5)
axs[0,0].scatter(cent[:,longS], cent[:,anchoS],  c="black", marker="D")
#axs[0,0].title("LongS - AnchoS")

#axs[0,1].title("LongS - LongP")
axs[0,1].scatter(pat[:,longS], pat[:,longP], c=color_scat, marker="o",s=5)
axs[0,1].scatter(cent[:,longS], cent[:,longP],  c="black", marker="D")

##axs[0,2].title("LongS - AnchoP")
axs[0,2].scatter(pat[:,longS], pat[:,anchoP], c=color_scat, marker="o",s=5)
axs[0,2].scatter(cent[:,longS], cent[:,anchoP],  c="black", marker="D")

#axs[1,0].title("AnchoS - LongP")
axs[1,0].scatter(pat[:,anchoS], pat[:,longP], c=color_scat, marker="o",s=5)
axs[1,0].scatter(cent[:,anchoS], cent[:,longP],  c="black", marker="D")

#axs[1,1].title("AnchoS - AnchoP")
axs[1,1].scatter(pat[:,anchoS], pat[:,anchoP], c=color_scat, marker="o",s=5)
axs[1,1].scatter(cent[:,anchoS], cent[:,anchoP],  c="black", marker="D")

#axs[1,2].title("LongP - AnchoP")
axs[1,2].scatter(pat[:,longP], pat[:,anchoP], c=color_scat, marker="o",s=5)
axs[1,2].scatter(cent[:,longP], cent[:,anchoP],  c="black", marker="D")'''

print("Clasificacion del K-medias",(pat[pat[:, -1] == 0]).shape[0],(pat[pat[:, -1] == 1]).shape[0],(pat[pat[:, -1] == 2]).shape[0])
print("Clasificacion correcta: [34 32 45]")

# Llamar a la función som_labels()
labels_som,Wx,Wy = som_labels()

# Usar las etiquetas generadas para análisis o comparación
print("Etiquetas generadas por SOM:", labels_som)
mat_cont = contingency_matrix(pat[:, -1], labels_som)
print("Matriz de contingencia :\n", mat_cont)

plt.show()