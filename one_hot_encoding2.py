import pandas as pd

cars = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\Machine_learning\\car_data.csv")

ohe_cars = pd.get_dummies(cars[['Car']])

print(ohe_cars.to_string())
