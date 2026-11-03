import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv(r"C:\Users\HPR\Desktop\ML Syllabus\Play Tennis.csv")

# Display dataset information
print("Number of rows:", len(df))
print("Dataset shape:", df.shape)
print(df.head())
print(df.tail())
print(df.describe(include='all'))

# Encode categorical data
string_to_int = preprocessing.LabelEncoder()
df = df.apply(string_to_int.fit_transform)

# Separate features and target
feature_cols = ['Outlook', 'Temprature', 'Humidity', 'Wind']
X = df[feature_cols]
y = df['Play_Tennis']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=100
)

# Create and train Decision Tree classifier
classifier = DecisionTreeClassifier(
    criterion="entropy",
    random_state=100
)

classifier.fit(X_train, y_train)

# Predict test data
y_pred = classifier.predict(X_test)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Compare actual and predicted values
data_p = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print(data_p)

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
