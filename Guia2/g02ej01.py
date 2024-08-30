import pandas as pd
import random
import numpy as np

def ej2(Estructura): #[2 1]
    df = pd.read_csv('Guia1\OR_trn.csv')
    arreglo = df.to_numpy()
    row= arreglo.shape[0]
    x = arreglo[: , :-1]  #Dejo solo las entradas en una matrix
    columna = np.full((arreglo.shape[0],1), -1) #Creo una columna de solo -1
    x = np.hstack((columna,x))
    yd = arreglo[:,-1] #Dejo los resultados esperados en un vector
    cantEntradas= x.shape[1] #Cant de columnas de la matriz
    Capas = Estructura.shape[0]
    ListaW = [] #Lista con las matrices de pesos para cada capa
    ListaE = []
    ListaDelta = []
    for capa in range(0,Capas):
        ListaW.append(np.random.uniform(low=-0.5,high=0.5,size=(Estructura[capa],cantEntradas)))
        ListaE.append(np.zeros((row,Estructura[capa]+1)))
        ListaDelta.append(np.zeros((row,Estructura[capa])))
   
    ListaE[0]=x
    ListaE.append(np.zeros((Estructura[-1],1))) #Agrego lugar para guardar la salida/s final/es
    epoca_max = 10
    err_max = 0.01
    k = 0.05
    for epoca in range(0,epoca_max):
        #PROPAGACION HACIA ADELANTE
        for capa in range(0,Capas):
            X = ListaE[capa] #Tengo las entradas a la capa
            W = ListaW[capa] #Tengo los W asociados a la capa
            for i in range(0,row):
                #y=ListaE[capa+1]
                #y[i]= sigmoide...
                ListaE[capa+1][i]= sigmoide((W * X[i].T).T) #MIRAR SI ASI SE CALCULA LA TRANSPUESTA #IMPLEMENTAR SIGMOIDE
                #AGREGARLE LOS -1
        #PROPAGACION HACIA ATRAS


        #ACTUALIZAR LOS PESOS
    return 0
ej2(np.array([2,1]))