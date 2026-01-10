import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

df = pandas.read_csv("C:\\Users\\Admin\\Downloads\\data.csv")

X = df[['Weight', 'Volume']]

X_scaled = scale.fit_transform(X)

print(X_scaled)
