import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
import schedule
import time

# Carica il file CSV in un DataFrame di Pandas
heart_data = pd.read_csv('heart.csv')
heart_data.drop_duplicates(inplace=True)

warnings.filterwarnings("ignore", category=UserWarning)

heart_data.isnull().sum()
heart_data.describe()
heart_data['target'].value_counts()

# Seleziona solo le features desiderate
selected_features = ["sex", "age", "cp", "thalach"]
X = heart_data[selected_features]
Y = heart_data['target']

# Splitting the Data into Training data & test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model Training, logistic Regression model
model = LogisticRegression(C=100, max_iter=5000, random_state=0, solver='newton-cg')

# Perform cross-validation
kfold = KFold(n_splits=10)
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")

# Print the results
msg = "Cross-Validation Accuracy: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

# Training the logistic regression model with training data
model.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on Training data:", training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on Test data:", test_data_accuracy)

# Build a predict system
input_data = (1, 41, 0, 158)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print('The Person does not have Heart Disease')
else:
    print('The Person has Heart Disease')

import joblib

# Salva il modello nel file modello_logistic_regression.pkl
with open('modello_logistic_regression.pkl', 'wb') as file:
    joblib.dump(model, file)


import pymongo
from sklearn.linear_model import LogisticRegression

import requests


def calculate_target_pbs():
    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    database = client["HeartFailure"]
    collection = database["Patients"]

    # Retrieve all patients from the collection
    patients = list(collection.find())

    # Load the trained logistic regression model
    # model = LogisticRegression(C=0.01, max_iter=5000, random_state=0, solver='liblinear')

    # Iterate over each patient and calculate target and PBS
    for patient in patients:
        input_data = [
            int(patient['age']), int(patient['sex']), int(patient['cp']),int(patient['thalach'])]

        # Convert input data to a numpy array and reshape for prediction
        input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Perform prediction using the logistic regression model
        prediction = model.predict(input_data_reshaped)

        # Update the patient's target value
        patient["id"] = int(patient["id"])  # Convert the ObjectId to a string
        patient["target"] = int(prediction[0])

        # Calculate PBS as the probability of target class 1
        prediction_proba = model.predict_proba(input_data_reshaped)
        pbs = round(prediction_proba[0][1] * 100, 2)
        patient["PBS"] = pbs

        # Make a POST request to update the patient's data in the database
        url = "http://localhost:5000/update_patient_data"  # Replace with your API endpoint URL
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=patient, headers=headers)

        if response.status_code == 200:
            print(f"Patient {patient['_id']} data updated successfully.")
        else:
            print(f"Failed to update data for patient {patient['_id']}.")

    # Close the MongoDB connection
    client.close()

# Schedule the function to run at 23:00 every evening
schedule.every().day.at("23:00").do(calculate_target_pbs)

# Keep the script running and continuously check the schedule
while True:
    schedule.run_pending()
    time.sleep(1)
