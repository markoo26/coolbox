from IPython.display import HTML
import pandas as pd
import base64

# function that takes in a dataframe and creates a text link to
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    """
        Generate a download link for a DataFrame as a CSV file.

        This function takes a DataFrame, converts it to a CSV format, encodes it in base64,
        and generates an HTML download link that allows the user to download the CSV file.

        Parameters:
        df (pandas.DataFrame): The DataFrame to be converted to CSV and downloaded.
        title (str): The text to display for the download link. Default is "Download CSV file".
        filename (str): The default filename for the downloaded CSV file. Default is "data.csv".

        Returns:
        IPython.core.display.HTML: An HTML object containing the download link.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame")

    try:
        csv = df.to_csv(index=False)  # Include index=False for a cleaner CSV file
        b64 = base64.b64encode(csv.encode()).decode()
        html = f'<a download="{filename}" href="data:text/csv;base64,{b64}" target="_blank">{title}</a>'
        return HTML(html)
    except Exception as e:
        raise RuntimeError(f"An error occurred while creating the download link: {e}")


def validate_kaggle_submission(submission_df, sample_submission_df):
    """
        Validates a Kaggle submission DataFrame against a sample submission DataFrame.

        This function checks for three key aspects to ensure that the submission DataFrame is valid:
        1. Column names match the sample submission.
        2. Data types of the columns match the sample submission.
        3. No missing values (NaNs) are present in the submission.

        Parameters:
        -----------
        submission_df : pandas.DataFrame
            The DataFrame containing the submission to be validated.
        sample_submission_df : pandas.DataFrame
            The DataFrame containing the sample submission for reference.

        Returns:
        --------
        bool
            True if all validation checks pass (i.e., columns match, dtypes match, and no NaNs present),
            False otherwise.

        Raises:
        -------
        AssertionError
            If any of the following checks fail:
            - Column names do not match.
            - Data types do not match.
            - Missing values are present in the submission.
        """

    columns_match = all(submission_df.columns == sample_submission_df.columns)
    dtypes_match = all(submission_df.dtypes == sample_submission_df.dtypes)
    no_missing_values = not submission_df.isna().any().any()

    assert columns_match, "Column mismatches between submissions"
    assert dtypes_match, "Data types mismatches between submissions"
    assert no_missing_values, "Nulls present in the submission"

    return {
        'columns_match': columns_match,
        'dtypes_match': dtypes_match,
        'no_missing_values': no_missing_values
    }
