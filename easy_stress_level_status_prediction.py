# We will predict the stress level using Decision tree
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

file_path = "synthetic_mental_health_dataset.csv"

mental_health_dataset = pd.read_csv(file_path)

mental_health_features = ['sleep_hours', 'screen_time', 'exercise_minutes']
X = mental_health_dataset[mental_health_features]

y = mental_health_dataset.stress_level

mental_health_model = DecisionTreeRegressor(random_state = 1)

mental_health_model.fit(X, y)

sleep_hours = float(input("Enter sleep hours per day"))

screen_time = float(input("Enter screen time (hours per day)"))

exercise_minutes = float(input("Enter exercise minutes per day"))

user_input = [[sleep_hours, screen_time, exercise_minutes]]

prediction = mental_health_model.predict(user_input)

print("Predicted Stress Level:", prediction[0])
