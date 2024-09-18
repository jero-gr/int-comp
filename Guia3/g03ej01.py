# Importar las librerías necesarias
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 

# Medidas de desempeño:
from sklearn.metrics import confusion_matrix            # Para generar matriz de confusión
from sklearn.metrics import ConfusionMatrixDisplay


#Usaremos el conjunto de datos Digits que viene con Scikit-learn. 
#Este conjunto de datos contiene imágenes de dígitos escritos a mano.

#Cargar el conjunto de datos digits
digits = load_digits()

#Separar las caracteristicas (X) y las etiquetas (y)
X = digits.data #X son las características (los valores de las imágenes).
Y=digits.target #y son las etiquetas (los dígitos correspondientes).

print(digits)

##Entrenamiento con una única partición (train_test_split)##
# División de los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train , X_test, Y_train , Y_test = train_test_split(X,Y,test_size=0.2)#,random_state=42) #Sacamos random state pq sino siempre genera los mismos aleatorios

#random_state=42 asegura que las divisiones aleatorias sean las mismas cada vez que ejecutes tu código.

#Clasificador de perceptron multicapa
#mlp = MLPClassifier(max_iter=1000,random_state=42) #Predeterminado 1 capa oculta 100 neurnonas, k = 0.001 con un max de 1000 iteraciones Funcion de activacion RELU
mlp = MLPClassifier(hidden_layer_sizes=(20,10),learning_rate_init=0.005, max_iter=1000, activation='logistic')#,random_state=42) #2capas ocultas de 20 y 10 k =0.005 Funcion de activacion Logistic


#Entrenar el modelo
mlp.fit(X_train,Y_train)


# Evaluar el modelo en los datos de prueba
score = mlp.score(X_test, Y_test)
print(f"Tasa de acierto con train_test_split: {score:.4f}")
print()



########################################CREAMOS KFOLD AHORA CON 5 PARTICIONES##################################################
# Crear el objeto KFold con 5 particiones
'''kf_5 = KFold(n_splits=5,shuffle=True,random_state=42)

print(kf_5)'''

# Calcular la media y varianza de los resultados
scores_5 = cross_val_score(mlp,X,Y,cv=5) #Datos Xdigits Ydigits #Divide en 5, 5 pruebas con 1 particion que es siempre diferente (entrena con la 1ra prueba 2,3,4,5)(entrena con 2da prueba 1,3,4,5)
mean_5 = np.mean(scores_5)
var_5 = np.var(scores_5)
print(f"Media de la tasa de acierto con 5 particiones (KFold): {mean_5:.4f}")
print(f"Varianza de la tasa de acierto con 5 particiones (KFold): {var_5:.4f}")

# Matriz para almacenar resultados de KFold
resultados_kfold = []
# Guardar resultados de las 5 particiones en la matriz
for i, score in enumerate(scores_5, start=1):
    resultados_kfold.append([i, score])  # Guardar partición y precisión

# Mostrar la matriz final
print("\nMatriz de resultados KFold (5 particiones):")
for row in resultados_kfold:
     print([row[0], float(row[1])])
print()

########################################CREAMOS KFOLD AHORA CON 10 PARTICIONES##################################################
# Crear el objeto KFold con 5 particiones
kf_10 = KFold(n_splits=10,shuffle=True,random_state=42)

# Calcular la media y varianza de los resultados
scores_10 = cross_val_score(mlp,X,Y,cv=kf_10) #Datos Xdigits Ydigits
mean_10 = np.mean(scores_10)
var_10 = np.var(scores_10)
print(f"Media de la tasa de acierto con 10 particiones (KFold): {mean_10:.4f}")
print(f"Varianza de la tasa de acierto con 10 particiones (KFold): {var_10:.4f}")

# Matriz para almacenar resultados de KFold
resultados_kfold = []
# Guardar resultados de las 5 particiones en la matriz
for i, score in enumerate(scores_10, start=1):
    resultados_kfold.append([i, score])  # Guardar partición y precisión

# Mostrar la matriz final
print("\nMatriz de resultados KFold (10 particiones):")
for row in resultados_kfold:
     print([row[0], float(row[1])])
print()
