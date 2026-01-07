import numpy
import matplotlib.pyplot as plt

X = numpy.random.normal(5.0, 1.0, 1000)

y = numpy.random.normal(10.0, 2.0, 1000)

plt.scatter(X, y)
plt.show()
