import pandas as pd

# Save filepath to a variable for easier access 
mental_health_filepath = "synthetic_mental_health_dataset.csv"

# Read the data and store in a dataframe titled mental_health_dataset
mental_health_dataset = pd.read_csv(mental_health_filepath)

# Print all columns from mental_health_dataset
print(mental_health_dataset.columns)
