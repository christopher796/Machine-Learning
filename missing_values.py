import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Save file path
file_path = "melb_data.csv"

# Read the data and store in dataframe called missing_dataset
missing_dataset = pd.read_csv(file_path)

# Target
y = missing_dataset.Price

melb_preds = missing_dataset.drop(['Price'], axis = 1)

# Features
X = melb_preds.select_dtypes(exclude=['object'])

# Train Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

# Function to compare different approaches to deal with missing values
def score_dataset(X_train, X_val, y_train, y_val):
    model = RandomForestRegressor(n_estimators = 10, random_state = 0)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)

# Scores From Approach 1(Drop columns with missing values)
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

#Drop columns in Training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis = 1)

reduced_X_val = X_val.drop(cols_with_missing, axis = 1)

print("MAE from Approach 1: ")
print(score_dataset(reduced_X_train, reduced_X_val, y_train, y_val))


# Scores from Approach 2 (Imputation)
my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_val = pd.DataFrame(my_imputer.transform(X_val))

# Put Back the removed column names
imputed_X_train.columns = X_train.columns
imputed_X_val.columns = X_val.columns

print("MAE from Approach 2 (Imputation): ")
print(score_dataset(imputed_X_train, imputed_X_val, y_train, y_val))


# Scores from Approach 3 (An Extension to Imputation)
# Make copy to avoid changing the original data
X_train_plus = X_train.copy()
X_val_plus = X_val.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_val_plus[col + '_was_missing'] = X_val_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_val_plus = pd.DataFrame(my_imputer.transform(X_val_plus))

print("MAE for Approach 3 (An Extension to imputation): ")
print(score_dataset(imputed_X_train_plus, imputed_X_val_plus, y_train, y_val))
