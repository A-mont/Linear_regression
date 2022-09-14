# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:37:39 2021

@author: monte
"""


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


   
df = pd.read_csv("FuelConsumption.csv") #lectura del dataset
#df.head()#LECTURA DEL TITULO DE CADA COLUMNA

#Podemos seleccionar algunas caracteristicas del dataset
cdf=df[["ENGINESIZE","CYLINDERS","CONSUMPTION","EMISSIONS"]]
#cdf=df.head()

#Creamos un histograma de las caracteristicas

#viz=cdf[["ENGINE SIZE","CYLINDERS","FUEL","EMISSIONS"]]
#viz.hist()
#plt.show()

 #Creamos un digrama de dispersion de las caracteristicas  
#plt.scatter(cdf.ENGINESIZE, cdf.EMISSIONS,  color='blue')
#plt.xlabel("ENGINE SIZE")
#plt.ylabel("Emission")
#plt.show()


 #CREAMOS UN SET DE ENTREMANIENTO Y DE PRUEBA(TRAIN/SPLIT).

msk=np.random.rand(len(df)) < 0.80
train=cdf[msk]#Set de entrenamiento
test=cdf[~msk]#set de prueba

#MODELO DE REGRESION SIMPLE

#### Entrenar distribución de los datos"
plt.scatter(train.ENGINESIZE, train.EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
#plt.show()

#MODELAMOS LA REGRESION LINEAL E IMPRIMIMOS COEFICIENTES DE REGRESION
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['EMISSIONS']])
regr.fit (train_x, train_y)

# Imprimimos los coeficientes de regresión.
print ('Coefficients:', regr.coef_)
print ('Intercept:',regr.intercept_)

#GENERAMOS UNA RECTA DE AJUSTE SOBRE LOS DATOS
plt.scatter(train.ENGINESIZE, train.EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")



#EVALUACIÓN DEL MODELO(OBTENEMOS COEFICIENTE DE CORRELACION)
from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['EMISSIONS']])
test_y_ = regr.predict(test_x)
print("Error medio absoluto: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Suma residual de los cuadrados (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


