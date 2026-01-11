import pandas as pd
from sklearn import linear_model

cars = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\Machine_learning\\car_data.csv")

ohe_cars = pd.get_dummies(cars[['Car']])

X = pd.concat([cars[['Volume', 'Weight']], ohe_cars], axis = 1)
y = cars['CO2']

regr = linear_model.LinearRegression()

regr.fit(X, y)

predictedCO2 = regr.predict([[2300, 1300, 0,0,0,0,0,0,0,1]])

print("The CO2 emmission of VW IS: ", predictedCO2)
