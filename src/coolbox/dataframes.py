import pandas as pd
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