import plotly.express as px
import optuna
def optuna_parallel_coordinates(optuna_study):
    """
        Generates and displays a parallel coordinates plot for an Optuna study.

        This function takes an Optuna study object, extracts the relevant parameters and objective values,
        and creates a parallel coordinates plot to visualize the relationship between these parameters
        and the objective value using Plotly.

        Parameters:
        study (optuna.study.Study): An Optuna study object containing the optimization trials.

        Returns:
        plotly.graph_objects.Figure: The generated parallel coordinates plot.
    """

    if not isinstance(optuna_study, optuna.study.Study):
        raise ValueError("The input 'study' must be an instance of optuna.study.Study")

    optuna_study_df = optuna_study.trials_dataframe()
    param_cols = [col for col in optuna_study_df.columns if 'params' in col or col == 'value']
    fig = px.parallel_coordinates(optuna_study_df, color="value",
                                  dimensions=param_cols,
                                  color_continuous_midpoint=10)

    return fig

