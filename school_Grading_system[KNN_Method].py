import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Training data
grade = [[10], [20], [30], [40], [50], [60], [70], [80], [90], [100]]
classes = [0 ,0, 0, 1, 1, 1, 1, 2, 2, 2]

# Model
knn = KNeighborsClassifier(n_neighbors= 3)
knn.fit(grade, classes)

# User input
grade_input = float(input("Enter the Student's grade:"))
new_grade = [[grade_input]]

# Prediction
prediction = knn.predict(new_grade)

# Output
if prediction == [0]:
    print("Poor Performer")

elif prediction == [1]:
    print("Average Performer")

else:
    print("Best Performer")

