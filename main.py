import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = 'healthcare_dataset.csv'
data = pd.read_csv(file_path)

# Step 2: Data Preprocessing
data = data.drop(columns=['Name', 'Doctor', 'Hospital', 'Insurance Provider', 'Date of Admission', 'Discharge Date'])

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
categorical_features = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Medication', 'Test Results']
le = LabelEncoder()
for feature in categorical_features:
    data[feature] = le.fit_transform(data[feature])

# Separate features and target
X = data.drop('Medical Condition', axis=1)
y = data['Medical Condition']

# Standardize numerical features
numerical_features = ['Age', 'Billing Amount', 'Room Number']
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Histograms for numerical features
X.hist(bins=30, figsize=(15, 10))
plt.show()

# Heatmap for correlations
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.show()


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestClassifier()
lr_model = LogisticRegression(max_iter=1000)

# Train models
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)


# Random Forest
rf_y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_y_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_y_pred))

# Logistic Regression
lr_y_pred = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_y_pred))
print("Logistic Regression Classification Report:\n", classification_report(y_test, lr_y_pred))


# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
best_rf_y_pred = best_rf_model.predict(X_test)
print("Optimized Random Forest Accuracy:", accuracy_score(y_test, best_rf_y_pred))
print("Optimized Random Forest Classification Report:\n", classification_report(y_test, best_rf_y_pred))

#Model Deployment
joblib.dump(best_rf_model, 'best_disease_prediction_model.pkl')


# (Additional code for monitoring can be added here)

# Flask API for predictions
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load('best_disease_prediction_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_df = pd.DataFrame(data, index=[0])
    data_df[numerical_features] = scaler.transform(data_df[numerical_features])
    data_df[categorical_features] = data_df[categorical_features].apply(lambda col: le.transform(col))
    prediction = model.predict(data_df)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
