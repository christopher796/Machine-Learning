import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

file_path = "synthetic_mental_health_dataset.csv"

mental_health_data = pd.read_csv(file_path)

features = ['sleep_hours', 'screen_time', 'exercise_minutes']

X = mental_health_data[features]

y = mental_health_data.stress_level

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d\t\tMean Absolute Error: %f" % (max_leaf_nodes, my_mae))
