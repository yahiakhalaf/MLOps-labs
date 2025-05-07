import yaml
import mlflow

def setup_mlflow():
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)

    tracking_uri = cfg['mlflow']['tracking_uri']
    experiment_name = cfg['mlflow']['experiment_name']

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
