import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

file_path = "melb_data.csv"

dataset = pd.read_csv(file_path)

# Select subsets of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = dataset[cols_to_use]

# Select Target
y = dataset.Price

my_pipeline = Pipeline(steps= [
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

# Multiply by -1 since sklearn calculates negative MAE

scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring = 'neg_mean_absolute_error')

print("Average MAE score (across experiment): ")
print(scores.mean())

