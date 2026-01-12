from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

X, y = datasets.load_iris(return_X_y= True)

clf = DecisionTreeClassifier(random_state=42)

sk_folds = StratifiedKFold(n_splits= 5)

scores = cross_val_score(clf, X, y, cv= sk_folds)

print("Stratified cross validation scores:", scores)

print("Average CV score:", scores.mean())

print("Number of CV scores used in Average:", len(scores))

