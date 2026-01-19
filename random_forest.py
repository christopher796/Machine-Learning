import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# store variable in a file path
file_path = "synthetic_mental_health_dataset.csv"

# Read the data and store it in a dataframe called mental_health_data
mental_health_data = pd.read_csv(file_path)

# Select features
features = ['sleep_hours', 'screen_time', 'exercise_minutes']

X = mental_health_data[features]

# Select the Target
y = mental_health_data.stress_level

# Splitting the data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Creating the model
mental_health_model = RandomForestRegressor(random_state = 1)

# Training the model
mental_health_model.fit(train_X, train_y)

# Making predictions on unseen Data
predictions = mental_health_model.predict(val_X)

# Output the mean absolute error
print(mean_absolute_error(val_y, predictions))
