# Importar las librerías necesarias
import pandas as pd
import random
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt

def A_g(epc):
    if epc<100:
        return 3
    elif epc<200:
        return 2
    elif epc<300:
        return 1
    else:
        return 0
        
def mu_fun(epc):
    if epc<100:
        return 0.8
    elif epc<200:
        return 2.2 + 0.007*float(epc)
    else:
        return 0.1

df = pd.read_csv("Guia4/circulo.csv")
mat_datos = df.to_numpy()
datos_row = mat_datos.shape[0] # Número de filas del conjunto de datos (número de patrones de entrenamiento)
datos_col = mat_datos.shape[1] # Número de filas del conjunto de datos (número de patrones de entrenamiento)

# Plotear cada par de puntos x_1, x_2
plt.figure(1)
for i in range(0, datos_row):
    plt.plot(mat_datos[i][0],mat_datos[i][1],marker='x',color="gray",fillstyle="none")

# Definir constantes
mu = 0.1
epc_max = 400

# Definir arquitectura (distancia Manhattan)
row = 5
col = 5
#vec = 2

mat_neu = [[0 for _ in range(col)] for _ in range(row)] # Matriz de neuronas
for i in range(0,row):
    for j in range(0,col):
        mat_neu[i][j] = np.random.uniform(low=-0.5,high=0.5,size=(1,datos_col)) # Dentro de cada neurona están los pesos

# Plotear la red neuronal
for i in range(0, row):
    for j in range(0, col):
        plt.plot(mat_neu[i][j][0][0],mat_neu[i][j][0][1],marker='o',color="blue",fillstyle="none")
        if i+1<row: plt.plot([mat_neu[i][j][0][0],mat_neu[i+1][j][0][0]],[mat_neu[i][j][0][1],mat_neu[i+1][j][0][1]],color="blue",linewidth=0.5)
        if j+1<col: plt.plot([mat_neu[i][j][0][0],mat_neu[i][j+1][0][0]],[mat_neu[i][j][0][1],mat_neu[i][j+1][0][1]],color="blue",linewidth=0.5)

for epc in range(0,epc_max):
    vec = A_g(epc)
    #mu = mu_fun(epc)
    print("Época " + str(epc) + " de " + str(epc_max))

    for xy in range(0,datos_row):
        # Obtener el índice de la neurona ganadora
        G_dif = -(mat_neu - mat_datos[xy][:])
        G_norm = np.zeros((row,col))
        for i in range(0,row):
            for j in range(0,col):
                G_norm[i][j] = np.linalg.norm(G_dif[i][j])

        argmin_G = np.argmin(G_norm)
        row_min = np.floor_divide(argmin_G,row)
        col_min = np.mod(argmin_G,col)

        # Actualizar los pesos
        it = 2*vec+1    # Cantidad de iteraciones dependiendo de la vecindad
        for i in range(0,vec):
            for j in range(-i,i+1):
                if row_min-vec+i>=0 and col_min+j>=0 and col_min+j<col:
                    mat_neu[row_min-vec+i][col_min+j] = mat_neu[row_min-vec+i][col_min+j] + mu * G_dif[row_min-vec+i][col_min+j]
                if row_min+vec-i<row and col_min+j>=0 and col_min+j<col:
                    mat_neu[row_min+vec-i][col_min+j] = mat_neu[row_min+vec-i][col_min+j] + mu * G_dif[row_min+vec-i][col_min+j]
        for i in range(-vec,vec+1):
            if col_min+i>=0 and col_min+i<col:
                mat_neu[row_min][col_min+i] = mat_neu[row_min][col_min+i] + mu * G_dif[row_min][col_min+i]
    
plt.figure(2)
# Plotear cada par de puntos x_1, x_2
for i in range(0, datos_row):
    plt.plot(mat_datos[i][0],mat_datos[i][1],marker='x',color="gray",fillstyle="none")

# Plotear la red neuronal
for i in range(0, row):
    for j in range(0, col):
        plt.plot(mat_neu[i][j][0][0],mat_neu[i][j][0][1],marker='o',color="blue",fillstyle="none")
        if i+1<row: plt.plot([mat_neu[i][j][0][0],mat_neu[i+1][j][0][0]],[mat_neu[i][j][0][1],mat_neu[i+1][j][0][1]],color="blue",linewidth=0.5)
        if j+1<col: plt.plot([mat_neu[i][j][0][0],mat_neu[i][j+1][0][0]],[mat_neu[i][j][0][1],mat_neu[i][j+1][0][1]],color="blue",linewidth=0.5)

plt.show()
    