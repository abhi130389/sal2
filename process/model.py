# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

os.chdir('D:\project_deployment\process')

#sal = pd.read_csv(r'D:\project_deployment\input\sal.csv',header = 0, index_col = None)
sal = pd.read_csv(r'input\sal.csv',header = 0, index_col = None)

X = sal[['X']]
Y = sal[['Y']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.25, random_state=10)

lm = LinearRegression()
lm.fit(X_train, Y_train)

#print('Intercept :', round(lm.intercept_, 2))
#print('Slope :', round(lm.coef_[0], 2))

from sklearn.metrics import mean_squared_error
Y_predict = lm.predict(X_test)
mse = mean_squared_error(Y_predict, Y_test)
print('MSE :', round(mse, 2))

filename = 'output/sal_model.pkl'
joblib.dump(lm, filename)
