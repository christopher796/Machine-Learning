import pandas as pd

# save file path to variable for easier access
mental_health_dataset = "synthetic_mental_health_dataset.csv"

# Read the data set and store in dataframe called mental_health_data
mental_health_data = pd.read_csv(mental_health_dataset)

#Print a summary of data in mental_health_data
mental_health_data.describe()

print(mental_health_data.describe())