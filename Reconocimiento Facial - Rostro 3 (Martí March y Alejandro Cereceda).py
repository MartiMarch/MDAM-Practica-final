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
y_3 = np.array(y_3, dtype=np.int16)

# Solución formato valido del array
y = np.ravel(y)
y_3 = np.ravel(y_3)

# Cantidad de datos usados para entrenar y para testear.
heldout = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Cantidad de veces que se repite
rounds = 10

# Variable utilizada para dibujar las curvas ROC
lw = 2

"""
# Porcentaje aleatorio que sera dedicado a entrenar y/o testear
seed = np.random.randint(100)
rng = np.random.RandomState(seed)
"""

#%% Clasificadores utilizados

# Contiene todos los clasificadores
wc = []

# Clasificador lineal
w = []
w.append(LinearDiscriminantAnalysis())
w.append("Linear Discriminant Analysis")
w.append("Yes")
wc.append(w)

# Clasificador cuadratico
w = []
w.append(QuadraticDiscriminantAnalysis())
w.append("Quadratic Discriminant Analysis")
w.append("Yes")
wc.append(w)

# Clasificador K vecinos
w = []
w.append(KNeighborsClassifier(n_neighbors = 4))
w.append("K Neighbors: K = 4")
w.append("No")
wc.append(w)

# Percepton
w = []
w.append(Perceptron(tol = 1e-4, max_iter = 1000000, eta0 = 1))
w.append("Percepton")
w.append("Yes")
wc.append(w)

# SGD
w = []
w.append(SGDClassifier(loss='squared_hinge', penalty = None, max_iter= 1000, tol = 1e-17))
w.append("SGD")
w.append("Yes")
wc.append(w)

# Calsificador red neuronal
w = []
w.append(MLPClassifier(hidden_layer_sizes=(3, 3), max_iter = 10000, alpha = 1e-4, solver = 'sgd', tol = 1e-4, random_state = 1, learning_rate_init = .1, early_stopping = True, validation_fraction = 0.1))
w.append("Red neuronal: 3x3 capas")
w.append("No")
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
    # Todas las medidas de precisión
    allAccTrain = []
    allAccTest = []
    
    
    print("\n-> [ " + w[1] + " ] <-\n");
    
    for i in heldout:
        
        # Precisión de todos los porcentajes
        AccTrain = []
        AccTest = []
        
        # Puntuaciones medias de todos los porcentajes
        ScoreTrain = []
        ScoreTest = []
        
        for r in range(rounds):
            #X_train, X_test, y_train, y_test = train_test_split(X_3, y_3, test_size = i, random_state = rng)
            
            # Se crean unos datos y etiquetas de forma equitativa a toda la base de datos
            X_train, X_test, y_train, y_test = train_test_split(X_3, y_3, test_size = i)
    
            # Se entrena al clasificador
            w[0].fit(X_train, y_train)
    
            # Se obtiene la precisión del entrenamiento para un porcentaje específico
            predTrain = w[0].predict(X_train)
        
            # Se obtiene la precisión del test para un porcentaje específico
            predTest = w[0].predict(X_test)    
            
            # La precisión se añade para porsteriormente dibujarse
            AccTrain.append(metrics.accuracy_score(y_train, predTrain))
            AccTest.append(metrics.accuracy_score(y_test, predTest))
            
            # Si se puede calcular la curva ROC de un clasificador se calula
            if  w[2] == "Yes":
                # Se obtiene la puntuación del entrenamiento para un porcentaje específico
                scoreTrain = w[0].decision_function(X_train)
                
                # Se obtiene la puntuación del test para un porcentaje específico
                scoreTest = w[0].decision_function(X_test)
                                      
                # Se almacenan las puntuaciones medias
                ScoreTrain.append(scoreTrain)
                ScoreTest.append(scoreTest)

        # Se saca la media de la precisión del entrenamiento
        sumaTrain = 0
        for n in AccTrain:
            sumaTrain = sumaTrain + n
        sumaTrain = sumaTrain / rounds
        
         # Se saca la media de la precisión de la prueba
        sumaTest = 0
        for n in AccTest:
            sumaTest = sumaTest + n
        sumaTest = sumaTest / rounds
        
        # Se saca la media de las predicciones del entrenamiento
        sumaScoreTrain = 0
        for n in ScoreTrain:
             sumaScoreTrain =  sumaScoreTrain + n
        sumaSacoreTrain = sumaScoreTrain / rounds
        
        # Se saca la media de las predicciones del entrenamiento
        sumaScoreTest = 0
        for n in ScoreTest:
            sumaScoreTest = sumaScoreTest + n
        sumaScoreTest = sumaScoreTest / rounds
        
        # Se guardan todas las medias de precisión, tanto de entrenamiento como de prueba
        allAccTrain.append(sumaTrain)
        allAccTest.append(sumaTest)
        
        print(f"Accuracy Train: {sumaTrain}, with {(int)((1-i)*100)}% of training data.")
        print(f"Accuracy Test: {sumaTest}, with {i*100}% of training data.")
        
        # Se calcula la matriz de confusión del test
        cfTrain = metrics.confusion_matrix(y_train, predTrain, labels=[0,1])
        print('Confusion matrix of Train:\n{}'.format(cfTrain))
        
        # Se calcula la matriz de confusión del train
        cfTest = metrics.confusion_matrix(y_test, predTest, labels=[0,1])
        print('Confusion matrix of Test:\n{}'.format(cfTest))
        
        # Se normaliza la matriz de confusión del train
        ncfTrain = cfTrain.astype('float') / cfTrain.sum(axis=1).reshape(2,1)
        print('Normalized confusion matrix of Train:\n{}'.format(ncfTrain))
        
        # Se normaliza la matriz de confusión del test
        ncfTest = cfTest.astype('float') / cfTest.sum(axis=1).reshape(2,1)
        print('Normalized confusion matrix of Test:\n{}'.format(ncfTest))
        
        # Se calcula la precision del train
        PrecisionTrain = metrics.precision_score(y_train, predTrain, pos_label=0)
        
        # Se calcula la precision del test
        PrecisionTest = metrics.precision_score(y_test, predTest, pos_label=0)
        
        # Se calula el recall del train
        RecallTrain = metrics.recall_score(y_train, predTrain, pos_label=0)
        
        # Se calula el recall del test
        RecallTest = metrics.recall_score(y_test, predTest, pos_label=0)
        
        # Se muestra la precisión y el recall del train
        print ('Precision and Recall of Train: {}, {}'.format(PrecisionTrain, RecallTrain))
        
        # Se muestra la precisión y el recall del test
        print ('Precision and Recall of Test: {}, {}\n'.format(PrecisionTest, RecallTest))
        
        # Si se puede calcular la curva ROC de un clasificador se calula
        if w[2] == "Yes":
            
            # Se obtiene los falsos positivos, los verdaderos positivos y los umbrales del entrenamiento para un porcentaje específico
            fprTrain, tprTrain, thresTrain = metrics.roc_curve(y_train, sumaScoreTrain, drop_intermediate=False)
            
            # Se dibuja la curva ROC para el entrenamiento
            plt.figure()
            plt.plot(fprTrain, tprTrain, color='darkorange', lw=lw, label='ROC curve (area = {:0.4f})'.format(metrics.auc(fprTrain, tprTrain)))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0+.01])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(str(w[1]) + ", heldout=" + str((int)((1-i)*100)) + "% of Train, 4 faces")
            plt.legend(loc="lower right")
            plt.show()
            
            # Se obtiene los falsos positivos, los verdaderos positivos y los umbrales del test para un porcentaje específico
            fprTest, tprTest, thresTest = metrics.roc_curve(y_test, sumaScoreTest, drop_intermediate=False)
            
            # Se dibuja la curva ROC para el test
            plt.figure()
            plt.plot(fprTest, tprTest, color='darkorange', lw=lw, label='ROC curve (area = {:0.4f})'.format(metrics.auc(fprTest,tprTest)))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0+.01])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(str(w[1]) + ", heldout=" + str((int)((1-i)*100)) + "% of Test, 4 faces")
            plt.legend(loc="lower right")
            plt.show()

    # Gráfica de precisión del entrenamiento y testeo para cada uno de los porcentajes de los datos usados
    plt.plot(1. - np.array(heldout), allAccTrain, '-o',lw = 1 ,label = '(Train)')
    plt.plot(1. - np.array(heldout), allAccTest, '-o',lw = 1 ,label = '(Test)')
    plt.legend(loc="lower right")
    plt.xlabel("Relative training set size")
    plt.ylabel("Accuracy")
    plt.title(w[1] + ", 4 faces")
    plt.show()
    
    # Se añaden los datos de precisión de todos los clasificadores para una futura gráfica
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
print("     Distinguiendo entre diez caras diferentes     ")
print("---------------------------------------------------")

for w in wc:
    # Todas las medidas de precisión
    allAccTrain = []
    allAccTest = []
    
    print("\n-> [ " + w[1] + " ] <-\n");
    
    for i in heldout:
        
        # Precisión de todos los porcentajes
        AccTrain = []
        AccTest = []
        
        # Puntuaciones medias de todos los porcentajes
        ScoreTrain = []
        ScoreTest = []
        
        for r in range(rounds):
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = i, random_state = rng)
            
            # Se crean unos datos y etiquetas de forma equitativa a toda la base de datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = i)
            
            # Se entrena al clasificador
            w[0].fit(X_train, y_train)
    
            # Se obtiene la precisión del entrenamiento para un porcentaje específico
            predTrain = w[0].predict(X_train)
            
            # Se obtiene la precisión del test para un porcentaje específico
            predTest = w[0].predict(X_test)    
            
            # La precisión se añade para porsteriormente dibujarse
            AccTrain.append(metrics.accuracy_score(y_train, predTrain))
            AccTest.append(metrics.accuracy_score(y_test, predTest))
            
            # Si se puede calcular la curva ROC de un clasificador se calula
            if  w[2] == "Yes":
                # Se obtiene la puntuación del entrenamiento para un porcentaje específico
                scoreTrain = w[0].decision_function(X_train)
                
                # Se obtiene la puntuación del test para un porcentaje específico
                scoreTest = w[0].decision_function(X_test)
                                      
                # Se almacenan las puntuaciones medias
                ScoreTrain.append(scoreTrain)
                ScoreTest.append(scoreTest)
        
        # Se saca la media de la precisión del entrenamiento
        sumaTrain = 0
        for n in AccTrain:
            sumaTrain = sumaTrain + n
        sumaTrain = sumaTrain / rounds
        
        # Se saca la media de la precisión de la prueba
        sumaTest = 0
        for n in AccTest:
            sumaTest = sumaTest + n
        sumaTest = sumaTest / rounds
        
        # Se saca la media de las predicciones del entrenamiento
        sumaScoreTrain = 0
        for n in ScoreTrain:
             sumaScoreTrain =  sumaScoreTrain + n
        sumaSacoreTrain = sumaScoreTrain / rounds
        
        # Se saca la media de las predicciones del entrenamiento
        sumaScoreTest = 0
        for n in ScoreTest:
            sumaScoreTest = sumaScoreTest + n
        sumaScoreTest = sumaScoreTest / rounds
        
        # Se guardan todas las medias de precisión, tanto de entrenamiento como de prueba
        allAccTrain.append(sumaTrain)
        allAccTest.append(sumaTest)
        
        print(f"Accuracy Train: {sumaTrain}, with {(int)((1-i)*100)}% of training data.")
        print(f"Accuracy Test: {sumaTest}, with {i}% of training data.")
        
        # Se calcula la matriz de confusión del test
        cfTrain = metrics.confusion_matrix(y_train, predTrain, labels=[0,1])
        print('Confusion matrix of Train:\n{}'.format(cfTrain))
        
        # Se calcula la matriz de confusión del train
        cfTest = metrics.confusion_matrix(y_test, predTest, labels=[0,1])
        print('Confusion matrix of Test:\n{}'.format(cfTest))
        
        # Se normaliza la matriz de confusión del train
        ncfTrain = cfTrain.astype('float') / cfTrain.sum(axis=1).reshape(2,1)
        print('Normalized confusion matrix of Train:\n{}'.format(ncfTrain))
        
        # Se normaliza la matriz de confusión del test
        ncfTest = cfTest.astype('float') / cfTest.sum(axis=1).reshape(2,1)
        print('Normalized confusion matrix of Test:\n{}'.format(ncfTest))
        
        # Se calcula la precision del train
        PrecisionTrain = metrics.precision_score(y_train, predTrain, pos_label=0)
        
        # Se calcula la precision del test
        PrecisionTest = metrics.precision_score(y_test, predTest, pos_label=0)
        
        # Se calula el recall del train
        RecallTrain = metrics.recall_score(y_train, predTrain, pos_label=0)
        
        # Se calula el recall del test
        RecallTest = metrics.recall_score(y_test, predTest, pos_label=0)
        
        # Se muestra la precisión y el recall del train
        print ('Precision and Recall of Train: {}, {}'.format(PrecisionTrain, RecallTrain))
        
        # Se muestra la precisión y el recall del test
        print ('Precision and Recall of Test: {}, {}\n'.format(PrecisionTest, RecallTest))
    
        # Si se puede calcular la curva ROC de un clasificador se calula
        if w[2] == "Yes":
            
            # Se obtiene los falsos positivos, los verdaderos positivos y los umbrales del entrenamiento para un porcentaje específico
            fprTrain, tprTrain, thresTrain = metrics.roc_curve(y_train, sumaScoreTrain, drop_intermediate=False)
            
            # Se dibuja la curva ROC para el entrenamiento
            plt.figure()
            plt.plot(fprTrain, tprTrain, color='darkorange', lw=lw, label='ROC curve (area = {:0.4f})'.format(metrics.auc(fprTrain, tprTrain)))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0+.01])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(str(w[1]) + ", heldout=" + str((int)((1-i)*100)) + "% of Train, 10 faces")
            plt.legend(loc="lower right")
            plt.show()
            
            # Se obtiene los falsos positivos, los verdaderos positivos y los umbrales del test para un porcentaje específico
            fprTest, tprTest, thresTest = metrics.roc_curve(y_test, sumaScoreTest, drop_intermediate=False)
            
            # Se dibuja la curva ROC para el test
            plt.figure()
            plt.plot(fprTest, tprTest, color='darkorange', lw=lw, label='ROC curve (area = {:0.4f})'.format(metrics.auc(fprTest,tprTest)))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0+.01])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(str(w[1]) + ", heldout=" + str((int)((1-i)*100)) + "% of Test, 10 faces")
            plt.legend(loc="lower right")
            plt.show()
    
    # Gráfica de precisión del entrenamiento y testeo para cada uno de los porcentajes de los datos usados
    plt.plot(1. - np.array(heldout), allAccTrain, '-o',lw = 1 ,label = '(Train)')
    plt.plot(1. - np.array(heldout), allAccTest, '-o',lw = 1 ,label = '(Test)')
    plt.legend(loc="lower right")
    plt.xlabel("Relative training set size")
    plt.ylabel("Accuracy")
    plt.title(w[1] + ", 10 faces")
    plt.show()
    
    # Se añaden los datos de precisión de todos los clasificadores para una futura gráfica
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