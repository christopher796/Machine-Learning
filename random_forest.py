import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

file_path = "synthetic_mental_health_dataset.csv"

mental_health_data = pd.read_csv(file_path)

features = ['sleep_hours', 'screen_time', 'exercise_minutes']

X = mental_health_data[features]
y = mental_health_data.stress_level

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

mental_health_model = RandomForestRegressor(random_state = 1)

mental_health_model.fit(train_X, train_y)

predictions = mental_health_model.predict(val_X)

print(mean_absolute_error(val_y, predictions))
