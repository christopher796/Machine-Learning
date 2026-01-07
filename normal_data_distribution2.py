import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(2.0, 10.0, 10000)

plt.hist(x, 100)
plt.show()
