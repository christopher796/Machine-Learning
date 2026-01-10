import numpy
from sklearn import linear_model

X = numpy.array([3.78,2.44,2.09,0.14,1.72,1.65,4.37,4.96,4.52,3.69,5.88]).reshape(-1,1)
y = numpy.array([0,0,0,0,0,0,1,1,1,1,1])

logr = linear_model.LogisticRegression()
logr.fit(X, y)

log_odds = logr.coef_

odds = numpy.exp(log_odds)

print(odds)
# The output (3.8) tells us that as the size increases by 1cm , the odds of the tumor being cancerous increases by 4X

