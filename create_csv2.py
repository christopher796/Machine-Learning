import pandas as pd

data = {
    "Car": ["Toyota", "Mitsubishi", "Scoda", "Fiat", "Mini", "VW", "Scoda", "Mercedes", "Ford"],
    "Model": ["Aygo", "Spacestar", "Citigo", "500", "Cooper", "Up!", "Fabia", "A-class", "Fiesta"],
    "Volume": [1000, 1200, 1000, 900, 1500, 1000, 1400, 1500, 1500],
    "Weight": [790, 1160, 929, 865, 1140, 929, 1109, 1365, 1112],
    "CO2": [99, 95, 90, 105, 105, 105, 90, 92, 98]
}

df = pd.DataFrame(data)
df.to_csv("car_data.csv", index=False)
