import plotly.graph_objs as go
import numpy as np

def make_html_filename(plot_title):
    """
        Setup a filename for .write_html method of Plotly.

        Parameters
        ----------
        plot_title : string
            User friendly plot name to be shown in Plotly figure

        Returns
        ---------

            Suitable, space-free title with .html extension
    """
    return plot_title.lower().replace(' ', '_') + '.html'


def calculate_boxplot_stats(input_series):
    """
       Compute and return boxplot statistics for a given series.

       Parameters:
       -----------
       input_series : array-like
           A numerical series (e.g., list, numpy array, pandas Series) for which boxplot statistics are calculated.

       Returns:
       --------
       go.Box
           A Plotly Box object with calculated statistics including Q1, median, Q3, whiskers, mean, and standard deviation.

       Notes:
       ------
       - The function calculates the 1st quartile (Q1), median, 3rd quartile (Q3), lower whisker, upper whisker, mean, and standard deviation.
       - The statistics are used to create a Plotly Box object for visualizing data distribution.
    """

    boxplot_stats = {
        'q1': np.percentile(input_series, 25),
        'median': np.median(input_series),
        'q3': np.percentile(input_series, 75),
        'lower_whisker': np.min(input_series),
        'upper_whisker': np.max(input_series),
        'mean': np.mean(input_series),
        'std_dev': np.std(input_series)

    }

    return go.Box(
        q1=[boxplot_stats['q1']],
        median=[boxplot_stats['median']],
        q3=[boxplot_stats['q3']],
        lowerfence=[boxplot_stats['lower_whisker']],
        upperfence=[boxplot_stats['upper_whisker']],
        mean=[boxplot_stats['mean']],  # Optional
        sd=[boxplot_stats['std_dev']],  # Optional
    )