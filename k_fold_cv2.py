from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score

X, y = datasets.load_iris(return_X_y = True)

clf = DecisionTreeClassifier(random_state= 42)

k_folds = KFold(n_splits = 4)

scores = cross_val_score(clf, X, y, cv=k_folds)

print("Cross Validation scores:", scores)

print("CV average score:", scores.mean())

print("Number of cross validation scores used in average:", len(scores))

