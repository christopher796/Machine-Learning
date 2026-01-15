import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Loading the dataset
file_path = "synthetic_mental_health_dataset.csv"
mental_health_dataset = pd.read_csv(file_path)

# Selecting features
features = ['sleep_hours', 'screen_time', 'exercise_minutes']
X = mental_health_dataset[features]

# Selecting the target output
y = mental_health_dataset.stress_level

# Splitting data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Creating the model
mental_health_model = DecisionTreeRegressor()

# Training the model
mental_health_model.fit(train_X, train_y)

# Making predictions on unseen data
val_predictions = mental_health_model.predict(val_X)

# Evaluating model performance
print(mean_absolute_error(val_y, val_predictions))