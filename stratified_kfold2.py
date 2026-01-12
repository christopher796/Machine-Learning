from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# load in and separate data
X, y = datasets.load_iris(return_X_y=True)

# Model for Evaluation
clf = DecisionTreeClassifier(random_state=42)

#Evaluating our model
sk_folds = StratifiedKFold(n_splits=4)
scores = cross_val_score(clf, X, y, cv=sk_folds)

# Display the scores
print("Cross Validation Scores:", scores)

print("Cross Validation scores Average:", scores.mean())

print("Number of CV scores used in average:", len(scores))
