import numpy
import matplotlib.pyplot as plt
from sklearn import metrics

actual = numpy.random.binomial(1, 0.7, size=500)
predicted = numpy.random.binomial(1, 0.7, size=500)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=[0, 1])
cm_display.plot()
plt.show()
