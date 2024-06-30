from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, laplacian_kernel


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
