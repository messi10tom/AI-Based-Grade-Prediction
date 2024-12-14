import pandas as pd 
from sklearn.preprocessing import StandardScaler


def encode_dataframe(dataframe: pd.DataFrame, columns: list = None, vocab: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Encodes specified columns of a DataFrame into numeric values based on their unique values.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to encode.
        columns (list): A list of column names to be encoded.
        vocab (dict, optional): A dictionary where each key is a column name and the value is the vocabulary (list of unique values).

    Returns:
        tuple: A tuple containing:
            - The updated DataFrame with specified columns encoded.
            - A dictionary where each key is a column name and the value is the vocabulary (list of unique values).
    """
    encoding_vocab = vocab if vocab is not None else {}

    for column in columns:
        if column in encoding_vocab:
            # Use the provided vocabulary for encoding
            unique_values = encoding_vocab[column]
        else:
            # Create a vocabulary (list of unique values) for the current column
            unique_values = dataframe[column].unique().tolist()
            encoding_vocab[column] = unique_values

        # Create a mapping from unique values to their indices
        value_to_index = {value: idx for idx, value in enumerate(unique_values)}

        # Apply the mapping to the column for encoding
        dataframe[column] = dataframe[column].map(value_to_index)

    return dataframe, encoding_vocab

def decode_dataframe(dataframe: pd.DataFrame, encoding_vocab: dict) -> pd.DataFrame:
    """
    Decodes specified columns of a DataFrame from numeric values to their original values.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to decode.
        encoding_vocab (dict): A dictionary where each key is a column name and the value is the vocabulary (list of unique values).

    Returns:
        pd.DataFrame: The updated DataFrame with specified columns decoded.
    """
    for column, vocab in encoding_vocab.items():
        # Create a mapping from indices to unique values
        index_to_value = {idx: value for idx, value in enumerate(vocab)}

        # Apply the mapping to the column for decoding
        dataframe[column] = dataframe[column].map(index_to_value)

    return dataframe


def normalize_data_zscore(df: pd.DataFrame, mean_dict: dict = None, std_dict: dict = None) -> tuple[pd.DataFrame, dict, dict]:
    """
    Normalize the given DataFrame using Z-score (Standardization).
    
    Parameters:
        df (pd.DataFrame): The input DataFrame to be standardized.
        mean_dict (dict, optional): A dictionary containing the mean of each column. If not provided, it will be calculated.
        std_dict (dict, optional): A dictionary containing the standard deviation of each column. If not provided, it will be calculated.
        
    Returns:
        tuple: 
            - pd.DataFrame: The standardized DataFrame.
            - dict: A dictionary containing the mean of each column.
            - dict: A dictionary containing the standard deviation of each column.
    """
    # Select columns to normalize (only numeric columns)
    columns_to_normalize = df.select_dtypes(include=['int64', 'float64']).columns

    if mean_dict is None or std_dict is None:
        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit the scaler and transform the data
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

        # Get mean and standard deviation for each column
        column_means = scaler.mean_
        column_stds = scaler.scale_

        # Convert to dictionaries with column names as keys
        mean_dict = {col: column_means[i] for i, col in enumerate(columns_to_normalize)}
        std_dict = {col: column_stds[i] for i, col in enumerate(columns_to_normalize)}
    else:
        # Use provided mean and std for normalization
        for col in columns_to_normalize:
            df[col] = (df[col] - mean_dict[col]) / std_dict[col]

    return df, mean_dict, std_dict

