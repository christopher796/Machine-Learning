import pandas as pd

mental_health_filepath = "synthetic_mental_health_dataset.csv"

mental_health_dataset = pd.read_csv(mental_health_filepath)

print(mental_health_dataset.columns)
