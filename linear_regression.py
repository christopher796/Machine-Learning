# linear_regresssion.py
#importing necessary libraries
# matplotlib.pyplot is used for plotting graphs
# scipy.stats is used for statistical functions including linear regression
import matplotlib.pyplot as plt
from scipy import stats

# Sample data points
# X represents the independent variables(input features)
#y represents the dependent variables(output/target)
X = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# Performing linear regression on data points
# slope: How much y changes when X increases by one unit
# intercept: The value of y when X is 0
# r: correlation coefficient(strength of relationship)
# p: statistical significance of the results
# std_err: Error estimate of the slope
slope, intercept, r, p, std_err = stats.linregress(X, y)

# Function to implement the regression equation and return the predicted y values
def myfunc(X):
    return slope * X + intercept

# Generating predicted y values using the regression function
mymodel = list(map(myfunc, X))

# Draws the actual data points
plt.scatter(X, y)
# Draws the regression line
plt.plot(X, mymodel)
# Displays the plot
plt.show()
