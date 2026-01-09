import numpy
from sklearn.metrics import r2_score

X = [1,3,5,7,9,10,12,14,16]
y = [50,60,65,70,72,75,78,80,85]

mymodel = numpy.poly1d(numpy.polyfit(X, y, 2))

speed = mymodel(10)

print(speed)
print(r2_score(y, mymodel(X)))
