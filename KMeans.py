import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = [4,5,10,4,3,11,14,6,10,12]
y = [21,19,24,17,16,25,24,22,21,21]

data = list(zip(X, y))

kmeans = KMeans(n_clusters = 2)
kmeans.fit(data)

plt.scatter(X, y, c= kmeans.labels_)
plt.show()
