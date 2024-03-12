# encode_factory.py

import pandas as pd
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer

def trace_padding(list_in: list, max_dim: int, default_value: str) -> list: 
    """
    Adds missing items to a list up to the specified maximum size (needed for simple-index encoding)
    
    Parameters:
    - list_in: list,
        the starting list
    - max_dim: int,
        the desired maximum size of the list
    - default_value: str,
        the value to be added as a missing element

    Returns:
    - A new list with missing items added
    """
    missing_elements = max_dim - len(list_in)
    list_padding = list_in + [default_value] * missing_elements
    return list_padding


def encoding_simple_index(traces):
    """
    Performs simple-index encoding
    
    Parameters:
    - traces: list,
        trace list

    Returns:
    - A DataFrame with encoded traces
    """

    traces_len = []
    for trace in traces:
        trace_parts = trace.split(" ")
        trace_len = len(trace_parts)
        traces_len.append(trace_len)

    max_trace_len = max(traces_len)

    print("Max trace lenght:", max_trace_len)

    col_names = [f'event_{j}' for j in range(max_trace_len)]

    print("Columns:", col_names)
        
    # Create an empty datafram
    out_df = pd.DataFrame(columns = col_names)
        
    # Add to each column the values in the list (separated)
    for trace in traces:
        trace_parts = trace.split(" ")
        # print(len(trace_parts))
        list_padding = trace_parts
        #if len(trace_parts) < max_trace_len:
        list_padding = trace_padding(trace_parts, max_trace_len, '')
        out_df.loc[len(out_df)] = list_padding

    # Endode binary of the simple-index inside the dataframe
    out_df = pd.get_dummies(out_df, columns=col_names) 
    return out_df

def encoding_binary(traces):
    """
    Performs binary encoding
    
    Parameters:
    - traces: list,
        trace list

    Returns:
    - A DataFrame with encoded traces
    """
    
    count_vect = CountVectorizer(lowercase=True)
    corpus = count_vect.fit_transform(traces)
    onehot = Binarizer().fit_transform(corpus.toarray())
    
    # out_df = pd.DataFrame(onehot, columns=[f'feature_{i}' for i in range(onehot.shape[1])]) # dataframe without column names
    out_df = pd.DataFrame(onehot, columns=count_vect.get_feature_names_out()) # dataframe with column names

    return out_df


def encoding_frequency(traces):
    """
    Performs frequency encoding
    
    Parameters:
    - traces: list,
        trace list

    Returns:
    - A DataFrame with encoded traces
    """
    
    model = CountVectorizer()
    encoding = model.fit_transform(traces)
    
    # out_df = pd.DataFrame(encoding.toarray(), columns=[f'feature_{i}' for i in range(onehot.shape[1])]) # dataframe without column names
    out_df = pd.DataFrame(encoding.toarray(), columns=model.get_feature_names_out()) # dataframe with column names
    
    return out_df
