import mlflow
from datetime import datetime

def mlflow_experiment(experiment_name, params):
    """
    A decorator to set up an MLFlow experiment, log parameters and metrics, and manage the run lifecycle.

    Args:
    - experiment_name (str): Name of the MLFlow experiment.
    - params (dict): Dictionary of parameters to log.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the MLFlow experiment
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run():
                # Log parameters
                for param, value in params.items():
                    mlflow.log_param(param, value)

                # Run the actual training function
                # Report the model and dict of metrics
                model, metrics = func(*args, **kwargs)

                # Log the metrics
                for metric, value in metrics.items():
                    if metric in metrics:
                        mlflow.log_metric(metric, value)

            # End the run automatically by exiting the `with` block
            return model, metrics

        return wrapper

    return decorator

def step_time_calculation(step_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            print(f"""{step_name.upper()} || Total execution time in seconds: {execution_time}""")
            return result
        return wrapper
    return decorator