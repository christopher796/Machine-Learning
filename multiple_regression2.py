import pandas
from sklearn import linear_model

df = pandas.read_csv("C:\\Users\\Admin\\Downloads\\data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict([[3300, 1500]])

print(predictedCO2)
