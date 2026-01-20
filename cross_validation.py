import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
# Save filepath
file_path = "melb_data.csv"

# Read data and store in dataset
dataset = pd.read_csv(file_path)

# Select subsets of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = dataset[cols_to_use]

# Select Target
y = dataset.Price

# Pipeline
my_pipeline = Pipeline(steps= [
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

# Multiply by -1 since sklearn calculates negative MAE

# Calculate the cross validation scores
scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring = 'neg_mean_absolute_error')

# Output
print("Average MAE score (across experiment): ")
print(scores.mean())

