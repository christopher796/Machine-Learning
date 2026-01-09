import numpy
import matplotlib.pyplot as plt

X = [1,2,3,4,5,6,7,8,9,10]
y = [3,5,7,9,11,13,15,17,19,21]

mymodel = numpy.poly1d(numpy.polyfit(X, y, 1))
myline = numpy.linspace(1, 10, 100)
plt.scatter(X, y)
plt.plot(myline, mymodel(myline))
plt.show()
