import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\Machine_learning\\comedy_show.csv")

d = {'UK': 0, 'USA': 1, 'N':2}
df['Nationality'] = df['Nationality'].map(d)

d = {'Yes': 1, 'No': 0}
df['Go'] = df['Go'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)

dtree.predict([[20, 3, 5, 2]])

if dtree.predict([[20, 3, 5, 2]]) == [1]:
    print("Attend the comedy show")

else:
    print("Do not Attend The Comedy Show")
    
