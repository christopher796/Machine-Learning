import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# An array that ressembles variables in a dataset
X = [4,5,10,4,3,11,14,8,10,12]
y = [21,19,24,17,16,25,24,22,21,21]
classes = [0,0,1,0,0,1,1,0,1,1]

# Turn the input Features into a set of points
data = list(zip(X, y))

# Fitting a KNN model on The Model Using 1 nearest neighbor
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(data, classes)

# Predict the class of new, unforeseen data points
new_X = 8
new_y = 21
new_point = [(new_X, new_y)]
prediction = knn.predict(new_point)

plt.scatter(X + [new_X], y + [new_y], c = classes + [prediction[0]])

plt.text(x = new_X-1.7, y = new_y-0.7, s = f"new point, class: {prediction[0]}")

plt.show()
