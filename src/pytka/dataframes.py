import pandas as pd
import plotly.express as px
import gc
def optimize_dataframe(df):
    """
        Optimize the memory usage of a DataFrame by downcasting numeric columns.

        This function prints the initial memory usage of the DataFrame, attempts to downcast
        numeric columns to more memory-efficient types, performs garbage collection,
        and then prints the final memory usage. The optimized DataFrame is returned.

        Parameters:
        df (pandas.DataFrame): The DataFrame to be optimized.

        Returns:
        pandas.DataFrame: The optimized DataFrame with reduced memory usage.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame")

    # Print initial memory usage
    initial_memory = df.memory_usage(deep=True).sum()
    print(f"Initial memory usage: {initial_memory / 1024 ** 2:.2f} MB")

    # Function to downcast data types
    def downcast_series(series, objects_to_categoricals=False):
        if pd.api.types.is_integer_dtype(series):
            return pd.to_numeric(series, downcast='integer')
        elif pd.api.types.is_float_dtype(series):
            return pd.to_numeric(series, downcast='float')
        elif pd.api.types.is_object_dtype(series) and objects_to_categoricals:
            num_unique_values = len(series.unique())
            num_total_values = len(series)
            if num_unique_values / num_total_values < 0.5:
                return series.astype('category')
        else:
            return series

    optimized_df = df.apply(downcast_series)

    # Perform garbage collection
    gc.collect()

    # Print final memory usage
    final_memory = optimized_df.memory_usage(deep=True).sum()
    print(f"Final memory usage: {final_memory / 1024 ** 2:.2f} MB")

    # Return the optimized DataFrame
    return optimized_df

def remove_dataframes(df_dict):
    """
    Delete DataFrames from the global namespace and display their memory usage.

    Parameters:
    -----------
    df_dict : dict
        A dictionary with DataFrame names as keys and pandas DataFrame objects as values.

    Prints:
    -------
    str
        The initial memory usage of each DataFrame in megabytes (MB).
        A message if a DataFrame is not present in the global namespace.

    Notes:
    ------
    - The function uses `del` to attempt to remove DataFrames from the global namespace.
    - It calls the garbage collector (`gc.collect()`) to release memory after each deletion.
    - Only DataFrames with names present in `globals()` can be deleted.
    """

    for df_name, df in df_dict.items():
        initial_memory = df.memory_usage(deep=True).sum()
        print(f"Initial memory usage of DataFrame {df_name}: {initial_memory / 1024 ** 2:.2f} MB")
        if df_name in globals():
            del df
        else:
            print(f"""DataFrame {df_name} not present in the runtime""")
        gc.collect()

def dummify_dataframe(pd_dataframe, dummy_columns, verbose=True):
    """
        Create dummies of pd_DataFrame for selected dummy_columns and remove
        the source columns.

        Parameters
        ----------
        pd_dataframe : pd.DataFrame()
           Underlying Pandas DataFrame

        dummy_columns : List[string]
           List of columns within `pd_dataframe` that should be processed

        verbose: Boolean
           Printout progress of dummification.
        Returns
        ---------
        Processed pd_dataframe with changed shape accordingly
    """

    for col in dummy_columns:
        if col in pd_dataframe.columns:
            if verbose:
                print(f'Processing feature {col}')
            #             pd_dataframe[col] = pd_dataframe[col].astype('int32') ##FIXME
            dummies = pd.get_dummies(pd_dataframe[col], prefix=col)
            pd_dataframe = pd.concat([pd_dataframe, dummies], axis=1)
            del pd_dataframe[col]
            gc.collect()
    return pd_dataframe


def remove_column_if_present(pd_dataframe, column):

    """
        Remove a specified column from a DataFrame if it exists.

        This function checks if the specified column is present in the DataFrame,
        and if so, removes it.

        Parameters:
        pd_dataframe (pandas.DataFrame): The DataFrame from which to remove the column.
        column (str): The name of the column to be removed.

        Returns:
        pandas.DataFrame: The modified DataFrame with the column removed if it existed.

        Raises:
        ValueError: If the input is not a pandas DataFrame or column name is not a string.
        """

    if not isinstance(pd_dataframe, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame")
    if not isinstance(column, str):
        raise ValueError("The column name must be a string")
    if column in pd_dataframe.columns:
        del pd_dataframe[column]


def text_input_to_numericals(dataframe, columns):
    """
        Convert text columns in a DataFrame to numerical values.

        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The DataFrame containing columns to convert.
        columns : list
            List of column names to be converted to numerical values.

        Returns:
        --------
        tuple
            A tuple containing the updated DataFrame and a dictionary of mappings used for conversion.
    """
    mappings = {}
    for column in columns:
        mapping_dict = {item_value: index for index, item_value in enumerate(dataframe[column].unique())}
        dataframe[column] = dataframe[column].map(mapping_dict)
        mappings[column] = mapping_dict
    return dataframe, mappings


def df_memory_usage(dataframes, show='mem_usage_perc', plot_height=1000, plot_width=1200):
    """
        Calculate and visualize memory usage of multiple pandas DataFrames.

        Parameters:
        -----------
        dataframes : dict
            A dictionary with DataFrame names as keys and DataFrames as values.
        show : str, optional
            Metric to display on the y-axis: 'mem_usage_perc' (default) or 'mem_usage'.
        plot_height : int, optional
            Plot height in pixels. Default is 1000.
        plot_width : int, optional
            Plot width in pixels. Default is 1200.

        Returns:
        --------
        pandas.DataFrame
            Summary of memory usage statistics for all DataFrames, including columns for
            'column', 'mem_usage' (MB), 'dtype', 'mem_usage_perc', and 'df_name'.
        """

    global_df_summary = pd.DataFrame(columns=['column', 'mem_usage', 'dtype', 'mem_usage_perc', 'df_name'])
    total_usage_dict = {}

    for df_name, df in dataframes.items():
        df_stats = df.memory_usage(deep=True).reset_index()
        df_stats.columns = ['column', 'mem_usage']
        df_stats['mem_usage'] = round(df_stats['mem_usage'] / (1024 ** 2), 2)

        dtypes = pd.DataFrame(df.dtypes).reset_index()
        dtypes.columns = ['column', 'dtype']

        df_stats = df_stats.merge(dtypes)
        df_stats['mem_usage_perc'] = round(df_stats['mem_usage'] / sum(df_stats['mem_usage']) * 100, 2)
        df_stats['df_name'] = df_name

        total_mem_usage_mb = df_stats['mem_usage'].sum()
        total_mem_usage_mb = round(total_mem_usage_mb, 2)

        total_usage_dict[df_name] = total_mem_usage_mb
        global_df_summary = pd.concat([global_df_summary, df_stats], axis=0)

    fig = px.bar(global_df_summary, x='column', y=show, color='dtype', facet_col='df_name',
                 facet_col_wrap=3, title=f'Total memory usage for runtime dataframes')

    fig.update_layout(height=plot_height, width=plot_width)
    for i, df_name in enumerate(dataframes.keys()):
        fig.layout.annotations[i]['text'] = f'Dataframe {df_name}: {total_usage_dict[df_name]} MB'

    fig.show()
    return global_df_summary
