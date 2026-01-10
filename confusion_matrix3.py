import numpy
import matplotlib.pyplot as plt
from sklearn import metrics

actual = numpy.random.binomial(1, 0.7, size=500)
predicted = numpy.random.binomial(1, 0.7, size=500)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

Accuracy = metrics.accuracy_score(actual, predicted)
Precision = metrics.precision_score(actual, predicted)
Recall = metrics.recall_score(actual, predicted)
Specificity = metrics.recall_score(actual, predicted, pos_label=0)
F_score = metrics.f1_score(actual, predicted)

print("Accuracy:", Accuracy)
print("Precision:", Precision)
print("Recall:", Recall)
print("Specificity:", Specificity)
print("F1 Score:", F_score)