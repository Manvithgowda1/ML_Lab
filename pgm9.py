import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Load the Iris dataset
iris = load_iris()

# Store features and target
X = iris.data
y = iris.target

# Count total samples
num_samples = X.shape[0]
print(f"Number of samples in the Iris dataset: {num_samples}")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Count training and testing samples
train_samples = X_train.shape[0]
test_samples = X_test.shape[0]

print(f"Number of samples in the training set: {train_samples}")
print(f"Number of samples in the testing set: {test_samples}")

# Create Random Forest classifier
rf = RandomForestClassifier(n_estimators=100)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Calculate accuracy
print("Random Forest model accuracy (in %):",
      metrics.accuracy_score(y_test, y_pred) * 100)

# Print actual and predicted values
print("Actual values:", y_test)
print("Predicted values:", y_pred)

# Display actual and predicted values in DataFrame
df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

print(df)

# Predict a new Iris flower
label_mapping = {
    0: "iris-setosa",
    1: "iris-versicolor",
    2: "iris-virginica"
}

y_pred = rf.predict([[3, 3, 2, 2]])

print("Result is:", label_mapping[y_pred[0]])
