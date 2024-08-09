import mlflow
from datetime import datetime
import tensorflow as tf


def list_devices():
    """
    List available GPUs and their memory usage.

    This function retrieves and prints the names and memory information of all physical GPU devices
    available to TensorFlow.

    Prints:
    -------
    str
        The name and memory usage (in bytes) of each GPU, formatted as 'GPU_name: {memory_info}'.
        If no GPUs are available, nothing is printed.

    Notes:
    ------
    - The function uses TensorFlow's `tf.config.list_physical_devices` to get a list of available GPUs.
    - Memory information is obtained using `tf.config.experimental.get_memory_info`,
      which returns a dictionary with 'current' and 'peak' memory usage.
    - This function assumes TensorFlow is properly installed and configured to detect GPU devices.
    """
    cpu_devices = tf.config.list_physical_devices('CPU')
    gpu_devices = tf.config.list_physical_devices('GPU')
    tpu_devices = tf.config.list_physical_devices('TPU')

    devices = {'CPU': cpu_devices, 'GPU': gpu_devices, 'TPU': tpu_devices}

    return devices
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
    """
        Decorator to measure and print the execution time of a function.

        Parameters:
        -----------
        step_name : str
            A name for the step being timed, used in log messages.

        Returns:
        --------
        function
            A wrapper function that prints execution start time, end time, and duration.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"""Started execution of {step_name.upper()} step""")
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            print(f"""{step_name.upper()} || Total execution time in seconds: {execution_time}""")
            return result
        return wrapper
    return decorator