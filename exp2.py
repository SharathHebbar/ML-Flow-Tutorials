import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import mlflow
import mlflow.sklearn
mlflow.tracking.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_name="Random_forest_diabetes_prediction")

df = pd.read_csv("https://raw.githubusercontent.com/SharathHebbar/Random-Forest/main/diabetes.csv")
df.shape

x = df.drop(['Outcome'], axis=1)
x.shape

y = df['Outcome']
y.shape

xtrain, xtest, ytrain, ytest = train_test_split(x, y, stratify=y, test_size=0.1, random_state=42)
xtrain.shape, xtest.shape, ytrain.shape, ytest.shape

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(xtrain, ytrain)


training_accuracy = rf.score(xtrain, ytrain)
testing_accuracy = rf.score(xtest, ytest) 

mlflow.start_run(run_name=("RF2"))
mlflow.log_metric("Accuracy", testing_accuracy)
mlflow.end_run()
