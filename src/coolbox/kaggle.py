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
