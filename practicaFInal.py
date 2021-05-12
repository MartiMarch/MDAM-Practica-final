#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Guardar la base de datos de Sickit-Learn en archivo MAT en caso de no existir

# Librerias
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

# Se cargar el MAT para distingir entre 10 personas
data = loadmat('myLFW.mat')
X = data['data']
y = data['nlab']

# Se muestra un mapa de los datos con diez clases, una por cara
plt.scatter(X[:,0], X[:,1], c = y)
plt.title("Mostrando las clases")
plt.show()

# Todas caras que no son la 3 se convierten en la clase 0. La cara 3 se pasa a ser la clase 1
cont = 0
for n in y:
    if n == 3:
        y[cont] = 1
    else:
        y[cont] = 0
    cont = cont + 1
    
# Se carga el MAT para distinguir entre 3 personas
X_3 = []
y_3 = []
cont = 0
while cont <= 199:
    X_3.append(X[cont])
    y_3.append(y[cont])
    cont = cont + 1
X_3 = np.array(X_3, dtype=np.float64)
y_3 = np.array(y_3, dtype=np.float64)

# Solución formato valido del array
y = np.ravel(y)
y_3 = np.ravel(y_3)

# Cantidad de datos usados para entrenar y para testear.
heldout = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Cantidad de veces que se repite
rounds = 10

# Porcentaje aleatorio que sera dedicado a entrenar y/o testear
seed = np.random.randint(100)
rng = np.random.RandomState(seed)

# Se muestra un mapa de los datos con solo dos clases
plt.scatter(X[:,0], X[:,1], c = y)
plt.title("Mostrando las clases")
plt.show()

#%% Clasificadores utilizados

# Contiene todos los clasificadores
wc = []

# Clasificador lineal
w = []
w.append(LinearDiscriminantAnalysis())
w.append("Linear Discriminant Analysis")
wc.append(w)

# Clasificador cuadratico
w = []
w.append(QuadraticDiscriminantAnalysis())
w.append("Quadratic Discriminant Analysis")
wc.append(w)

# Clasificador K vecinos
w = []
w.append(KNeighborsClassifier(n_neighbors = 10))
w.append("K Neighbors: K = 10")
wc.append(w)

# Percepton
w = []
w.append(Perceptron(tol = 1e-5, max_iter = 100, eta0 = 1))
w.append("Percepton")
wc.append(w)

# SGD
w = []
w.append(SGDClassifier(loss='squared_hinge', penalty=None))
w.append("SGD")
wc.append(w)

# Calsificador red neuronal
w = []
w.append(MLPClassifier(hidden_layer_sizes=(5), max_iter = 200, alpha = 1e-4, solver = 'sgd', tol = 1e-4, random_state=  1, learning_rate_init = .1))
w.append("Red neuronal: 5 capas")
wc.append(w)

# Calsificador red neuronal
w = []
w.append(MLPClassifier(hidden_layer_sizes=(10), max_iter = 200, alpha = 1e-4, solver = 'sgd', tol = 1e-4, random_state=  1, learning_rate_init = .1))
w.append("Red neuronal: 10 capas")
wc.append(w)

# Calsificador red neuronal
w = []
w.append(MLPClassifier(hidden_layer_sizes=(10, 10), max_iter = 200, alpha = 1e-4, solver = 'sgd', tol = 1e-4, random_state=  1, learning_rate_init = .1))
w.append("Red neuronal: 10x10 capas")
wc.append(w)

#%% a) discriminate your person from other 3 persons out of the ten available (you choose), and 

# Porcentaje de precision de todos los test
AllTests = []

# Porcentaje de precision de todos los entrenes
AllTrain = []

print("---------------------------------------------------")
print("    Distinguiendo entre cuatro caras diferentes    ")
print("---------------------------------------------------")

for w in wc:
    allAccTrain = []
    allAccTest = []
    print("\n-> [ " + w[1] + " ] <-\n");
    for i in heldout:
        AccTrain = []
        AccTest = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = train_test_split(X_3, y_3, test_size = i, random_state = rng)
    
            w[0].fit(X_train, y_train)
    
            predTrain = w[0].predict(X_train)
            AccTrain.append(metrics.accuracy_score(y_train, predTrain))
        
            predTest = w[0].predict(X_test)    
            AccTest.append(metrics.accuracy_score(y_test, predTest))

        sumaTrain = 0
        for n in AccTrain:
            sumaTrain = sumaTrain + n
        sumaTrain = sumaTrain / rounds
        
        sumaTest = 0
        for n in AccTest:
            sumaTest = sumaTest + n
        sumaTest = sumaTest / rounds
        
        """
        # Nos quedamos con el dataset en el que la cantidad de datos para entrenar es menor
        if i == 0.9:
            plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
            plt.title("Images after apply clasifier " + str(w[1]) + ", 4 faces")
            plt.show()
        """    
        
        allAccTrain.append(sumaTrain)
        allAccTest.append(sumaTest)
        print(f"Accuracy Train: {sumaTrain}, with {(int)((1-i)*100)}% of training data.")
        print(f"Accuracy Test: {sumaTest}, with {(int)((1-i)*100)}% of training data.\n")

    plt.plot(1. - np.array(heldout), allAccTrain, '-o',lw = 1 ,label = '(Train)')
    plt.plot(1. - np.array(heldout), allAccTest, '-o',lw = 1 ,label = '(Test)')
    plt.legend(loc="lower right")
    plt.xlabel("Relative training set size")
    plt.ylabel("Accuracy")
    plt.title(w[1] + ", 4 faces")
    plt.show()
    
    AllTests.append(w[1])
    AllTests.append(allAccTest)
    AllTrain.append(w[1])
    AllTrain.append(allAccTrain)

# Se dibujan los resultados de todos los test
size = len(AllTests)
cont = 0
while cont < size:
    plt.plot(1. - np.array(heldout), AllTests[cont + 1], '-o', lw = 1 , label = AllTests[cont])
    cont = cont + 2
plt.legend(loc="lower right")
plt.xlabel("Relative training set size")
plt.ylabel("Accuracy")
plt.title("All clasifiers accuracy tests, 4 faces")
plt.show()  

# Se dibujan los resultados de todos los entrenes
size = len(AllTrain)
cont = 0
while cont < size:
    plt.plot(1. - np.array(heldout), AllTrain[cont + 1], '-o', lw = 1  ,label = AllTrain[cont])
    cont = cont + 2
plt.legend(loc="lower right")
plt.xlabel("Relative training set size")
plt.ylabel("Accuracy")
plt.title("All clasifiers accuracy trainings, 4 faces")
plt.show()  

#%% b) discriminate your person from all the remaining ones.

# Porcentaje de precision de todos los test
AllTests = []

# Porcentaje de precision de todos los entrenes
AllTrain = []

print("---------------------------------------------------")
print("    Distinguiendo entre diez caras diferentes    ")
print("---------------------------------------------------")

for w in wc:
    allAccTrain = []
    allAccTest = []
    print("\n-> [ " + w[1] + " ] <-\n");
    for i in heldout:
        AccTrain = []
        AccTest = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = i, random_state = rng)
    
            w[0].fit(X_train, y_train)
    
            predTrain = w[0].predict(X_train)
            AccTrain.append(metrics.accuracy_score(y_train, predTrain))
        
            predTest = w[0].predict(X_test)    
            AccTest.append(metrics.accuracy_score(y_test, predTest))
         
        """
        # Nos quedamos con el dataset en el que la cantidad de datos para entrenar es menor
        if i == 0.9:
            plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
            plt.title("Images after apply clasifier " + str(w[1]) + ", 10 faces")
            plt.show()
        """
            
        sumaTrain = 0
        for n in AccTrain:
            sumaTrain = sumaTrain + n
        sumaTrain = sumaTrain / rounds
        
        sumaTest = 0
        for n in AccTest:
            sumaTest = sumaTest + n
        sumaTest = sumaTest / rounds
        
        allAccTrain.append(sumaTrain)
        allAccTest.append(sumaTest)
        print(f"Accuracy Train: {sumaTrain}, with {(int)((1-i)*100)}% of training data.")
        print(f"Accuracy Test: {sumaTest}, with {(int)((1-i)*100)}% of training data.\n")
    
    plt.plot(1. - np.array(heldout), allAccTrain, '-o',lw = 1 ,label = '(Train)')
    plt.plot(1. - np.array(heldout), allAccTest, '-o',lw = 1 ,label = '(Test)')
    plt.legend(loc="lower right")
    plt.xlabel("Relative training set size")
    plt.ylabel("Accuracy")
    plt.title(w[1] + ", 10 faces")
    plt.show()
    
    AllTests.append(w[1])
    AllTests.append(allAccTest)
    AllTrain.append(w[1])
    AllTrain.append(allAccTrain)
    
# Se dibujan los resultados de todos los test
size = len(AllTests)
cont = 0
while cont < size:
    plt.plot(1. - np.array(heldout), AllTests[cont + 1], '-o', lw = 1 , label = AllTests[cont])
    cont = cont + 2
plt.legend(loc="lower right")
plt.xlabel("Relative training set size")
plt.ylabel("Accuracy")
plt.title("All clasifiers accuracy tests, 10 faces")
plt.show()  

# Se dibujan los resultados de todos los entrenes
size = len(AllTrain)
cont = 0
while cont < size:
    plt.plot(1. - np.array(heldout), AllTrain[cont + 1], '-o', lw = 1 , label = AllTrain[cont])
    cont = cont + 2
plt.legend(loc="lower right")
plt.xlabel("Relative training set size")
plt.ylabel("Accuracy")
plt.title("All clasifiers accuracy trainings, 10 faces")
plt.show()   