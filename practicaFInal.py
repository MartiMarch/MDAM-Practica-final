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

# Se cargar el MAT
data = loadmat('myLFW.mat')
X = data['data']
y = data['nlab']
    
cont = 0
for n in y:
    if n == 3:
        y[cont] = 1
    else:
        y[cont] = 0
    cont = cont + 1
    
# Solución formato valido del array
y = np.ravel(y)

# Cantidad de datos usados para entrenar y para testear.
heldout = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Cantidad de veces que se repite
rounds = 10

# Porcentaje aleatorio que sera dedicado a entrenar y/o testear
seed = np.random.randint(100)
rng = np.random.RandomState(seed)

#%% Clasificador linear, cuadratico, K vecinos, red neuronal, percepton, SGD y redes neuronales

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
    plt.title(w[1])
    plt.show()