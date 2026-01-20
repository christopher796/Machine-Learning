import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
# Save file path
file_path = "melb_data.csv"

# Read data and store in dataset
dataset = pd.read_csv(file_path)
# Select features
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

X = dataset[cols_to_use]
# Target
y = dataset.Price
# Train Test Splt
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Define model
my_model = XGBRegressor(n_estimators = 500, learning_rate = 0.05, random_state=42, early_stopping_rounds=30, n_jobs = 5)
# Fit model
my_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
# Predict
predictions = my_model.predict(X_valid)
# Output
print("Mean Absolute Error: ", mean_absolute_error(predictions, y_valid))
