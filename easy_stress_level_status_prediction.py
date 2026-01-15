# We will predict the stress level using Decision tree
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Store the variable in a file path
file_path = "synthetic_mental_health_dataset.csv"

# Read the data and store in a dataframe called mental_health dataset
mental_health_dataset = pd.read_csv(file_path)

# Select features
mental_health_features = ['sleep_hours', 'screen_time', 'exercise_minutes']
X = mental_health_dataset[mental_health_features]

y = mental_health_dataset.stress_level

# Define model
mental_health_model = DecisionTreeRegressor(random_state = 1)

# Fit model
mental_health_model.fit(X, y)

# Ask user for feature values
sleep_hours = float(input("Enter sleep hours per day: "))
screen_time = float(input("Enter screen time (hours per day): "))
exercise_minutes = float(input("Enter exercise minutes per day: "))

# Create input in the correct shape
user_input = [[sleep_hours, screen_time, exercise_minutes]]

# Predict The stress level
prediction = mental_health_model.predict(user_input)

# Output the stress level
print("Predicted Stress Level:", prediction[0])
