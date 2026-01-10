import numpy
import matplotlib.pyplot as plt

numpy.random.seed()

X = numpy.random.normal(4, 1, 100)
y = numpy.random.normal(160, 50, 100)/ X

train_X = X[:80]
train_y = y[:80]

test_X = X[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_X, train_y, 4))

myline = numpy.linspace(0, 8, 100)

plt.scatter(train_X, train_y)
plt.plot(myline, mymodel(myline))
plt.show()
