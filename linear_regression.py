import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model  #Linear Model for Linear regression

#Dataset need to be read by read_csv
dataset = pd.read_csv("G:\Dataset\FuelConsumption.csv")

#process dataset to specific column
 
processed_dataset = dataset[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# Regression model
regression = linear_model.LinearRegression()
x = np.asanyarray(processed_dataset[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(processed_dataset[['CO2EMISSIONS']])

# Fit x and y with regression model
regression.fit (x, y)

# print the coefficients of the model 
print ('Coefficients: ', regression.coef_)