import numpy
from sklearn import linear_model

X = numpy.array([3.78,2.44,2.09,0.14]).reshape(-1,1)
y = numpy.array([0,0,1,1])

logr = linear_model.LinearRegression()
logr.fit(X, y)

def logit2prob(logr, X):
    log_odds = logr.coef_ * X + logr.intercept_
    odds = numpy.exp(log_odds)
    probability = odds / (1 + odds)
    return(probability)

print(logit2prob(logr, X))
