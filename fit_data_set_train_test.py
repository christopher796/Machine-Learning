import numpy
import matplotlib.pyplot as plt

numpy.random.seed(2)

X = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100)/ X

train_X = X[:80]
train_y = y[:80]

test_X = X[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_X, train_y, 4))

myline = numpy.linspace(0, 6, 100)

plt.scatter(train_X, train_y)
plt.plot(myline, mymodel(myline))
plt.show()
