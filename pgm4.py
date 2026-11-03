import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv(r"C:\Users\HPR\Desktop\ML Syllabus\Play Tennis.csv")

# Dataset information
print(len(df))
print(df.shape)
print(df.head())
print(df.tail())
print(df.describe(include='all'))

# Encode categorical data
string_to_int = preprocessing.LabelEncoder()
df = df.apply(string_to_int.fit_transform)
print(df)

# Separate features and target
feature_cols = ['Outlook', 'Temprature', 'Humidity', 'Wind']
X = df[feature_cols]
y = df.Play_Tennis

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=100
)

# Train Decision Tree using Gini criterion
classifier = DecisionTreeClassifier(
    criterion="gini",
    random_state=100
)

classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Actual vs Predicted
data_p = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print(data_p)

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Classification Report
print(classification_report(y_test, y_pred, zero_division=0))
