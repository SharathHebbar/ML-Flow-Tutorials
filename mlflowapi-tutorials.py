import mlflow
mlflow.tracking.set_tracking_uri("http://127.0.0.1:5000/")

print(mlflow.tracking.get_tracking_uri())

experiment_name = mlflow.get_experiment("8a9962918fce45ae9169d17c146de04a")
print(experiment_name)