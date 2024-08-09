from nltk.corpus import words
import pandas as pd
import re
import time
import nltk
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from .features import extend_features_with_similarities_and_distances
import string
from datetime import datetime

nltk.download('words')


def string_to_lowercase_word_list(input_string):
    """
        Convert a string into a list of lowercase words.

        This function removes punctuation from the input string, splits the string into words,
        and converts each word to lowercase.

        Parameters:
        input_string (str): The input string to be converted.

        Returns:
        list: A list of lowercase words from the input string.

        Example:
        >>> string_to_lowercase_word_list("Bonjour, ceci est une phrase de test.")
        ['bonjour', 'ceci', 'est', 'une', 'phrase', 'de', 'test']
    """

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    no_punctuation_string = input_string.translate(translator)

    # Split the string into words
    words = no_punctuation_string.split()

    # Convert words to lowercase
    lowercase_words = [word.lower() for word in words]

    return lowercase_words

def calculate_english_word_ratio(prompt):
    """
        Calculate the ratio of English words in a given prompt.

        This function takes a prompt (a string) and calculates the ratio of words in the prompt
        that are recognized as English words.

        Parameters:
        prompt (str): The input string to be analyzed.

        Returns:
        float: The ratio of English words to the total number of words in the prompt.

        Raises:
        ValueError: If the prompt is not a string or if the prompt is empty.
    """

    end_of_sentence_chars = ['?','.','!']

    # Check if there are more than one punctuation chars in single sentence

    punctuation_counts = [prompt.count(punctuation) > 1 for punctuation in end_of_sentence_chars]

    if not prompt[-1] in end_of_sentence_chars:
        raise ValueError("Improper sentence - it does not end with end of sentence characters")

    elif any(punctuation_counts):
        raise ValueError("Improper sentence - it some end of sentence characters are present more than once")

    words_set = set(words.words())
    prompt_words = string_to_lowercase_word_list(prompt)
    english_words_count = sum(word in words_set for word in prompt_words)
    return english_words_count / len(prompt_words)


def avg_word_length(sentence):
    """
        Calculate the average length of words in a given sentence.

        This function takes a sentence, splits it into words, and calculates the average
        length of the words. If the sentence is empty or contains no words, it returns 0.

        Parameters:
        sentence (str): The input sentence to be analyzed.

        Returns:
        float: The average length of the words in the sentence. Returns 0 if the sentence is empty.
    """
    words = string_to_lowercase_word_list(sentence)
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)


def avg_sentence_word_length(text):
    """
       Calculate the average word length for each sentence in a given text.

       This function takes a text, splits it into sentences, and calculates the average
       word length for each sentence. It then returns the average of these sentence
       average word lengths. If the text contains no sentences, it returns 0.

       Parameters:
       text (str): The input text to be analyzed.

       Returns:
       float: The average word length per sentence. Returns 0 if the text contains no sentences.

       Raises:
       ValueError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("The input must be a string")

    sentences = re.split(r'[.!?]', text)
    sentence_lengths = [avg_word_length(sentence) for sentence in sentences if sentence]

    if len(sentence_lengths) == 0:
        return 0
    return sum(sentence_lengths) / len(sentence_lengths)


def tfidf_vectorize_sequences(datasets, tfidf_vectorizer_params, fields_with_sequences, svd_n_components, process_fn=None):
    start_time = time.perf_counter()
    print(f"Started TFIDF vectorization at: {str(datetime.now())}")
    assert len(datasets) == 2, "Not two datasets [train|test] were provided, please check"
    assert len(datasets[1]) < len(datasets[0]), "The test dataset is bigger than the training set, \
                                                please swap in function calling"

    initial_train_data = datasets[0]
    initial_test_data = datasets[1]

    tfidf_vectorizer = TfidfVectorizer(**tfidf_vectorizer_params)

    # Combine train and test data into a single DataFrame
    full_data = pd.concat([initial_train_data, initial_test_data], ignore_index=True)

    # Clean and prepare the text columns if applicable
    if process_fn:
        for field in fields_with_sequences:
            full_data[field] = full_data[field].apply(process_fn)

    full_corpus = pd.concat([full_data[field] for field in fields_with_sequences], ignore_index=True)

    # Compute the TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(full_corpus)
    print('TFIDF vectorizer fitted successfully')

    # Perform dimensionality reduction with Truncated Singular Value Decomposition
    svd = TruncatedSVD(n_components=svd_n_components, random_state=42)
    reduced_matrix = svd.fit_transform(tfidf_matrix)

    print("TruncatedSVD applied properly")

    # Split into sequence fields
    splitted_full_tfidf = {}
    splitted_set_tfidf = {}
    training_length = len(initial_train_data)

    for field_index, field in enumerate(fields_with_sequences):
        # Store separate field in separate dict key
        splitted_full_tfidf[field] = reduced_matrix[len(full_data) * field_index:len(full_data) * (field_index + 1)]
        # Divide it further to store for each fields array for training and test
        splitted_set_tfidf[f"train_tfidf_{field}"] = splitted_full_tfidf[field][:training_length]
        splitted_set_tfidf[f"test_tfidf_{field}"] = splitted_full_tfidf[field][training_length:]

    # Create DataFrames to hold the SVD features for train and test sets
    # feature_names = [f'svd_feature_{i}' for i in range(svd_n_components)]
    train_features = pd.DataFrame(index=initial_train_data.index)
    test_features = pd.DataFrame(index=initial_test_data.index)

    # Assign SVD features to the respective columns in the feature DataFrames
    for i in range(svd_n_components):
        for field in fields_with_sequences:
            train_features[f'svd_{field}_{i}'] = splitted_set_tfidf[f"train_tfidf_{field}"][:, i]
            test_features[f'svd_{field}_{i}'] = splitted_set_tfidf[f"test_tfidf_{field}"][:, i]

    # Extend features by adding distances and similarities
    train_features, test_features = extend_features_with_similarities_and_distances(train_features=train_features,
                                                                                    test_features=test_features,
                                                                                    reduced_matrix=reduced_matrix)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f'Vectorization execution time: {execution_time}s')
    return train_features, test_features, reduced_matrix, tfidf_matrix, tfidf_vectorizer


def describe_tfidf_vectorizer(tfidf_vectorizer, tfidf_matrix, perc_words=5, perc_docs=5,
                              starting_word_index=0):
    """
    Generate a preview of the TF-IDF matrix for a given TF-IDF vectorizer.

    This function takes a TF-IDF vectorizer and generates a preview of the TF-IDF matrix,
    showing a sample of documents and words based on the specified percentages. It prints
    the total number of words, the initial and truncated matrix shapes, and provides
    examples of document/word pairs.

    Parameters:
    tfidf_vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): The fitted TF-IDF vectorizer.
    perc_words (int): The percentage of words to include in the preview. Default is 5%.
    perc_docs (int): The percentage of documents to include in the preview. Default is 5%.
    starting_word_index (int): The starting index of words for the preview. Default is 15000.

    Returns:
    pandas.DataFrame: A DataFrame containing the truncated TF-IDF matrix preview.

    Raises:
    ValueError: If the input percentages are not between 0 and 100.
    """
    if not (0 <= perc_words <= 100) or not (0 <= perc_docs <= 100):
        raise ValueError("Percentage values must be between 0 and 100")

    all_words = tfidf_vectorizer.get_feature_names_out()

    num_words = int(perc_words / 100 * len(all_words))
    num_docs = int(perc_docs / 100 * tfidf_matrix.shape[0])
    print(f"Total words: {len(all_words)}")
    print(f"Initial matrix shape: {tfidf_matrix.shape}")

    preview_tfidf = pd.DataFrame(tfidf_matrix[:num_docs, starting_word_index:starting_word_index + num_words].todense())
    print(f"Truncated matrix shape: {preview_tfidf.shape}")
    preview_tfidf.columns = all_words[starting_word_index:starting_word_index + num_words]

    print('\nExamples of (document/word) pairs:\n')
    for item in tfidf_matrix:
        print(item)
        break

    return preview_tfidf
