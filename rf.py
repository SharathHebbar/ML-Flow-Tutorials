import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn
mlflow.tracking.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_name="Random_forest_diabetes_prediction")

## Auto Logging
mlflow.sklearn.autolog()
print("Start")
with mlflow.start_run(run_name="random-forest-demo-1") as run:

    df = pd.read_csv("https://raw.githubusercontent.com/SharathHebbar/Random-Forest/main/diabetes.csv")
    df.shape

    x = df.drop(['Outcome'], axis=1)
    x.shape

    y = df['Outcome']
    y.shape

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, stratify=y, test_size=0.1, random_state=42)
    xtrain.shape, xtest.shape, ytrain.shape, ytest.shape

    from sklearn.ensemble import RandomForestClassifier
    # Hparams
    n_estimators = 150
    criterion = "gini"
    max_depth = 10
    min_samples_split = 4
    max_features = "sqrt"
    n_jobs = 2
    random_state = 24
    verbose = 1
    class_weight = "balanced_subsample"
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        class_weight=class_weight
    )
    params = {
        "n_estimators" : 150,
        "criterion" : "gini",
        "max_depth" : 10,
        "min_samples_split" : 4,
        "max_features" : "sqrt",
        "n_jobs" : 2,
        "random_state" : 24,
        "verbose" : 1,
        "class_weight" : "balanced_subsample"}

    # ML Flow log params
    # mlflow.log_params(params)
    rf.fit(xtrain, ytrain)

    training_accuracy = rf.score(xtrain, ytrain)
    testing_accuracy = rf.score(xtest, ytest) 
    y_pred = rf.predict(xtest)


    # Logging metric
    
    mlflow.set_tag("classifier", "rf")
    
print("Done")