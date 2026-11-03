import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

# Load dataset
df = pd.read_csv(r"C:\Users\HPR\Desktop\ML Syllabus\Play_Tennis_reg.csv")

# Dataset information
print(len(df))
print(df.shape)

# Select features and target
X = df.drop("Golf Players", axis=1)
y = df["Golf Players"]

print(X)
print(y)

# Encode categorical features
string_to_int = LabelEncoder()
X = X.apply(string_to_int.fit_transform)

print(X)

# Train Decision Tree Regressor
reg = DecisionTreeRegressor()
reg = reg.fit(X, y)

# Predict results
y_pred = reg.predict(pd.DataFrame(
    [[2, 1, 0, 1]],
    columns=X.columns
))
print("Result is:", y_pred)

y_pred = reg.predict(pd.DataFrame(
    [[2, 1, 0, 0]],
    columns=X.columns
))
print("Result is:", y_pred)

y_pred = reg.predict(pd.DataFrame(
    [[1, 2, 0, 0]],
    columns=X.columns
))
print("Result is:", y_pred)
