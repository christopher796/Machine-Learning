import numpy
import matplotlib.pyplot as plt

X = [1,3,5,7,9,10,12,14,16]
y = [50,60,65,70,72,75,78,80,85]

mymodel = numpy.poly1d(numpy.polyfit(X, y, 2))
myline = numpy.linspace(1, 16, 100)
plt.scatter(X, y)
plt.plot(myline, mymodel(myline))
plt.show()
