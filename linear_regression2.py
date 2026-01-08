import matplotlib.pyplot as plt
from scipy import stats

X = [9,7,3,8,1]
y = [ 10,6,8,9,5]

slope, intercept, r, p, std_error = stats.linregress(X, y)

def myfunc(X):
    return slope * X + intercept

mymodel = list(map(myfunc, X))

plt.scatter(X, y)
plt.plot(X, mymodel)
plt.show()

