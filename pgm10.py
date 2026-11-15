import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Read the dataset
data = pd.read_csv("Iris.csv")

# Display dataset shape
print(data.shape)

# Display first five rows
print(data.head())

# Drop unnecessary Id column
data = data.drop('Id', axis=1)

# Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Shape of X is %s and shape of y is %s" % (X.shape, y.shape))

# Count unique species
total_classes = y.nunique()
print("Number of unique species in dataset are:", total_classes)

# Display class distribution
distribution = y.value_counts()
print(distribution)

# Split dataset into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create AdaBoost classifier
adb = AdaBoostClassifier()

# Train the model
adb_model = adb.fit(X_train, Y_train)

# Calculate validation accuracy
print("The accuracy of the model on validation set is", adb_model.score(X_val, Y_val))

# Make predictions
y_pred = adb_model.predict(X_val)

# Calculate prediction accuracy
accuracy = accuracy_score(Y_val, y_pred)
print(f"The Accuracy of Prediction on Iris Flower is: {accuracy}")

# Display actual and predicted values
df = pd.DataFrame({
    'Actual': Y_val,
    'Predicted': y_pred
})

print(df)
