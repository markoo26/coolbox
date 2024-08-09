from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, laplacian_kernel
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, ClusterCentroids, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.express as px


def extend_features_with_similarities_and_distances(train_features, test_features, reduced_matrix):
    for df, len_df in zip([train_features, test_features], [len(train_features), len(test_features)]):
        prompt_indices = df.index
        # Calculate cosine similarity features
        df['similarity_pa'], df['similarity_pb'] = zip(*[
            calculate_cosine_similarity(reduced_matrix, i, i + len_df, i + 2 * len_df)
            for i in prompt_indices
        ])
        # Calculate Euclidean distance features
        df['euclidean_pa'], df['euclidean_pb'] = zip(*[
            calculate_distances(reduced_matrix, i, i + len_df, i + 2 * len_df,
                                euclidean_distances)
            for i in prompt_indices
        ])
        # Calculate Laplacian kernel distance features
        df['laplacian_pa'], df['laplacian_pb'] = zip(*[
            calculate_distances(reduced_matrix, i, i + len_df, i + 2 * len_df,
                                laplacian_kernel)
            for i in prompt_indices
        ])

    return train_features, test_features


def calculate_cosine_similarity(tfidf_matrix,
                                prompt_idx,
                                response_a_idx,
                                response_b_idx):
    # Cosine similarity between prompt (p) and response_a (a)
    similarity_pa = cosine_similarity(
        tfidf_matrix[prompt_idx].reshape(1, -1),
        tfidf_matrix[response_a_idx].reshape(1, -1)
    )[0][0]

    # Cosine similarity between prompt (p) and response_b (b)
    similarity_pb = cosine_similarity(
        tfidf_matrix[prompt_idx].reshape(1, -1),
        tfidf_matrix[response_b_idx].reshape(1, -1)
    )[0][0]

    return similarity_pa, similarity_pb


def calculate_distances(tfidf_matrix,
                        prompt_idx,
                        response_a_idx,
                        response_b_idx,
                        distance_metric):
    # Distance between prompt (p) and response_a (a)
    distance_pa = distance_metric(
        tfidf_matrix[prompt_idx].reshape(1, -1),
        tfidf_matrix[response_a_idx].reshape(1, -1)
    )[0][0]

    # Distance between prompt (p) and response_b (b)
    distance_pb = distance_metric(
        tfidf_matrix[prompt_idx].reshape(1, -1),
        tfidf_matrix[response_b_idx].reshape(1, -1)
    )[0][0]

    return distance_pa, distance_pb


def imbalanced_resampling(X, y, method='RandomUnderSampler', verbose=False, **kwargs):

    choices = {'RandomUnderSampler': RandomUnderSampler(**kwargs),
               'NearMiss': NearMiss(**kwargs),
               'TomekLinks': TomekLinks(**kwargs),
               'ClusterCentroids': ClusterCentroids(**kwargs),
               'EditedNearestNeighbors': EditedNearestNeighbours(**kwargs),
               'SMOTE': SMOTE(**kwargs),
               'ADASYN': ADASYN(**kwargs)}

    X_resampled_train, y_resampled_train = choices[method].fit_resample(X, y)

    if verbose:
        print(f"""X_under_train.shape: {X_resampled_train.shape}""")
        print(f"""X.shape: {X.shape}""")
        print(f"""y_under_train.shape: {y_resampled_train.shape}""")
        print(f"""y.shape: {y.shape}""")


        print("Undersampling validation")

        print(pd.Series(y_resampled_train).value_counts())

    return X_resampled_train, y_resampled_train

def filter_outliers(df, column, ratio, precision=2):

    """
        Filter outliers from a specified column in a DataFrame using the Interquartile Range (IQR) method.

        This function identifies and removes outliers from a given column based on a specified IQR ratio.
        It calculates the lower and upper bounds for valid data and filters out values outside this range.
        The percentage of records removed is also printed.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the data to filter.
        column : str
            The column name in the DataFrame from which to filter outliers.
        ratio : float
            The multiplier for the IQR to determine the cutoff for outliers.
        precision : int, optional (default=2)
            The number of decimal places to round the calculated bounds.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame with outliers removed from the specified column.

        Prints:
        -------
        str
            The range of values considered non-outliers for the given ratio.
        str
            The number and percentage of records removed from the DataFrame.
        """


    Q3 = df[column].quantile(0.75)
    Q1 = df[column].quantile(0.25)

    IQR = Q3 - Q1
    lower_bound = round(Q1 - ratio * IQR, precision)
    upper_bound = round(Q3 + ratio * IQR, precision)

    print(f'Column: {column} valid range including IQR +/- ({ratio} * STD): <{lower_bound}, {upper_bound}>')

    data = df[column]
    outside_range_mask = (data < lower_bound) | (data > upper_bound)

    records_before = len(df)
    df = df[~outside_range_mask]
    records_after = len(df)

    print(
        f"""Removed {records_before - records_after} records, which is {round((records_before - records_after) / records_before * 100, 2)}%""")
    return df

def quick_pca(X, y):
    """
        Perform PCA and DBSCAN clustering on the given dataset and visualize the results.

        Parameters:
        -----------
        X : array-like
            The input data to be transformed using PCA.
        y : array-like
            The target variable (not used in this function, but typically for classification tasks).

        Prints:
        -------
        str
            The explained variance ratio of the first two principal components.

        Visualizes:
        -----------
        Plotly Scatter Plot
            A scatter plot of the PCA-transformed data, highlighting DBSCAN clusters and outliers.

        Notes:
        ------
        - PCA is performed with two components to reduce dimensionality.
        - DBSCAN is used to cluster the PCA-transformed data and identify outliers.
        """

    # Perform PCA with two components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(data=X_pca, columns=['pca1', 'pca2'])

    # Append target, change to string to enable Plotly filtering out of traces
    pca_df['target'] = y.astype('string')

    # Print the explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f'Explained variance by component 1: {explained_variance[0]:.4f}')
    print(f'Explained variance by component 2: {explained_variance[1]:.4f}')

    # Fit DBSCAN to the PCA-transformed data
    dbscan = DBSCAN(eps=0.4, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_pca)

    # Add the DBSCAN cluster labels to the DataFrame
    pca_df['cluster'] = dbscan_labels

    # Highlight outliers with a different color
    pca_df['outlier'] = np.where(dbscan_labels == -1, 'Outlier', 'Inlier')

    # Plot the PCA results with clusters and outliers
    fig = px.scatter(
        pca_df,
        x='pca1',
        y='pca2',
        color='target',
        symbol='outlier',
        title='PCA of X with DBSCAN Clustering and Outliers Highlighted',
        labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
        hover_data=['cluster']
    )

    fig.show()
