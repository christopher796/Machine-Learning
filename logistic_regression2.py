import numpy
from sklearn import linear_model

X = numpy.array([3.78, 2.44, 2.09, 0.14]).reshape(-1, 1)
y = numpy.array([0,0,1,1])

logr = linear_model.LogisticRegression()
logr.fit(X, y)

predicted = logr.predict(numpy.array([3.5]).reshape(-1,1))

if predicted == 0:
    print("The tumor is not cancerous")

else:
    print("The tumor is cancerous")

    
