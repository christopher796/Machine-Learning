import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = [12,10,8,9,6,14,15,13,7,11]
y = [29,20,24,22,27,25,26,24,28,23]

data = list(zip(X, y))

inertias = []

for i in range (1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker = 'o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
