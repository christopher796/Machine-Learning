import numpy
import matplotlib.pyplot as plt

numpy.random.seed(2)

X = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100)/ X

train_X = X[:80]
train_y = y[:80]

test_X = X[80:]
test_y = y[80:]

plt.scatter(train_X, train_y)
plt.show()

plt.scatter(test_X, test_y)
plt.show()
