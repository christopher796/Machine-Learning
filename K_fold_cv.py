from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score

#load in and separate data
X, y = datasets.load_iris(return_X_y = True)

# A model for evaluation
clf = DecisionTreeClassifier(random_state = 42)

# Evaluating our Model
k_folds = KFold(n_splits = 5)

# performance stored on scores
scores = cross_val_score(clf, X, y, cv = k_folds)

print("Cross Validation Scores:", scores)

print("Average CV score:", scores.mean())

print("Number of CV scores used in Average:", len(scores))

