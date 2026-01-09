import numpy
from sklearn.metrics import r2_score

X = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22]
y = [100,90,80,60,55,60,65,70,70,75,76,78,79,90,99,99,100,101,102]

mymodel = numpy.poly1d(numpy.polyfit(X, y, 3))

print(r2_score(y, mymodel(X)))