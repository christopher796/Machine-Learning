import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = [12,10,8,9,6,14,15,13,7,11]
y = [29,20,24,22,27,25,26,24,28,23]

data = list(zip(X, y))

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(X, y, c=kmeans.labels_)
plt.show()
