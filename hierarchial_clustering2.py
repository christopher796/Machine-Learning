import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

X = [4,5,10,4,3,11,14,6,10,12]
y = [21,19,24,17,16,25,24,22,21,21]

data = list(zip(X,y))

hierarchial_cluster = AgglomerativeClustering(n_clusters=4, linkage='ward')

labels = hierarchial_cluster.fit_predict(data)

plt.scatter(X, y, c=labels)
plt.show()
