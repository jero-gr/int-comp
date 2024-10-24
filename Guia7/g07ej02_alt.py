import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Funciones
def prob_ij(D,tabu,alpha,beta,sigma,i):
    eta_row = np.divide(1,D[i,:])
    eta_row[i] = 0
    sigma_row = sigma[i,:]
    p_num = np.multiply(np.pow(sigma_row,alpha),np.pow(eta_row,beta))
    p_den = np.sum(np.multiply(np.pow(np.multiply(sigma_row,tabu),alpha),np.pow(np.multiply(eta_row,tabu),beta)))
    p = np.multiply(np.divide(p_num,p_den),tabu)
    p = np.nan_to_num(p)
    return np.cumsum(p)

def select_node(p,tabu):
    p_rand = np.random.random()
    for i in range(0,len(p)):
        if p_rand <= p[i] and tabu[i]==1:
            return i

def ants_equal(ant_path):
    for i in range(1,len(ant_path)):
        if not (np.array_equiv(ant_path[i-1],ant_path[i])):
            return False
    return True

def dist_equal(ant_dist):
    for i in range(1,len(ant_dist)):
        if ant_dist[i-1] != ant_dist[i]:
            return False
    return True

# Importar datos
df = pd.read_csv("Guia7/gr17.csv",header=None)
D = df.to_numpy()
D_row = len(D)

# Sistema de hormigas

# Constantes y variables
N = 30      # Cantidad de hormigas
origen = 0    # Nodo origen
it_max = 1500
rho = 0.5
Q = 1
metodo = 'L' # G=global, U=uniforme, L=local

# 1. En t=0 se inicializan las feromonas con valores pequeños al azar
sigma_0 = 0.5
sigma = np.random.random((D_row,D_row))*sigma_0

# 2. Se ubican las N hormigas en el nodo origen.
ant_path = []
for ant in range(0,N):
    ant_path.append(np.array([origen]))

ant_dist = np.zeros(N)

# 3. Repetir hasta que todas las hormigas sigan el mismo camino:
for it in range(0,it_max):

    print('Iteración ' + str(it) + ', sum(sigma)=' + str(np.sum(sigma)))

#   3.1. Para cada hormiga
    for k in range(0,N):

        # El camino está vacío
        path = np.array([origen])
        tabu = np.ones(D_row)
        tabu[origen] = 0

        # Todavía no se llegó a destino
        destino_fin = False

#       3.1.2. Repetir hasta completar el camino
        while (destino_fin == False):

            # Guardo en i el nodo actual y lo califico como tabú para evitar ciclos
            i = path[-1]
            tabu[i] = 0

#           3.1.2.1. Seleccionar el próximo nodo según la probabilidad
            p_ij = prob_ij(D,tabu,1,1,sigma,i)
            j = select_node(p_ij,tabu)

#           3.1.2.2. Agregar un paso (i,j) al camino.
            path = np.append(path,j)

            # Si todas las ciudades fueron visitadas, se cierra el camino
            if (len(path)==D_row):
                destino_fin = True
                path = np.append(path,origen)

#       3.1.3. Calcular la longitud del camino encontrado
        path_dist = 0
        for step in range(1,len(path)):
            path_dist = path_dist + D[path[step-1],path[step]]
        
        # Almaceno el camino y su distancia
        ant_path[k] = np.copy(path)
        ant_dist[k] = np.copy(path_dist)

#   3.2. Para cada conexión (i,j)
#   3.2.1. Reducir por evaporación la cantidad de feromonas
    sigma = sigma * (1-rho)

    delta_sigma = np.zeros_like(sigma)

    for k in range(0,N):    # Para cada hormiga
        path = np.copy(ant_path[k])  # Uso su camino
        path_dist = np.copy(ant_dist[k]) # Y la distancia ya calculada

        for step in range(1,len(path)): # Para cada paso del camino
            i=path[step-1]  # Tomo los índices
            j=path[step]

            # Dependiendo del método modifico el sigma
            if metodo=='G':
                delta_sigma[i,j] += Q/path_dist
            if metodo=='U':
                delta_sigma[i,j] += Q
            if metodo=='L':
                delta_sigma[i,j] += Q/D[i,j]

#   3.2.2. Depositar feromonas proporcionalmente a la bondad de la solución
    sigma += delta_sigma

    # Si todas las hormigas tienen el mismo camino, 

    if dist_equal(ant_dist) == True:
        print('Iteración ' + str(it))
        print('Se llegó al camino.')
        break

# 4. Devolver el camino más corto
indx = np.argmin(ant_dist)
print(ant_path[indx])
print(ant_dist[indx])