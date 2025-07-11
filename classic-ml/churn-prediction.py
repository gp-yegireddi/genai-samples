# churn_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Create or Load Dataset
def create_dataset():
    data = {
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'Age': np.random.randint(18, 70, 100),
        'Tenure': np.random.randint(1, 10, 100),
        'Balance': np.random.uniform(1000, 100000, 100),
        'NumOfProducts': np.random.randint(1, 4, 100),
        'HasCrCard': np.random.randint(0, 2, 100),
        'IsActiveMember': np.random.randint(0, 2, 100),
        'EstimatedSalary': np.random.uniform(20000, 120000, 100),
        'Churn': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)
    return df

# 2. Preprocessing
def preprocess_data(df):
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])  # Male=1, Female=0

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Train the Model
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# 4. Evaluate the Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Predict for a New Customer
def predict_new(model):
    new_data = [[1, 35, 5, 35000, 2, 1, 1, 50000]]  # Example input
    prediction = model.predict(new_data)
    print("\nPrediction for new customer (1 = Churn, 0 = Stay):", prediction[0])

# --------- Run All Steps ---------
if __name__ == "__main__":
    df = create_dataset()
    print("Sample Dataset:\n", df.head())

    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)
    predict_new(model)
