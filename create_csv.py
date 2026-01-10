import pandas as pd

data = {
    "Age": [36,42,23,52,43,44,66,35,52,35,24,18,48],
    "Experience": [10,12,4,4,21,14,3,14,13,5,3,3,9],
    "Rank": [9,4,6,4,8,5,7,9,7,9,5,7,9],
    "Nationality": ["UK","USA","N","USA","USA","UK","N","UK","N","N","USA","UK","UK"],
    "Go": ["No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes"]
}

df = pd.DataFrame(data)
df.to_csv("comedy_show.csv", index=False)
