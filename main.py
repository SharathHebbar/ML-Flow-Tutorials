import mlflow
import mlflow.sklearn
print("ml flow imported")

from mlflow.deployments import get_deploy_client, run_local

run_local(target="127.0.0.1:5000", name="MLDeploy", model_uri="mlartifacts\\669619307338469755\\6b30fb4d6d834353b840b621c08ec678\\artifacts\\model")