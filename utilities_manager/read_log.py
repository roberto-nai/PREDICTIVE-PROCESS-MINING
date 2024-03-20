import pandas as pd
import os

def read_log(dir_name: str, file_name: str, separator: str, replace_space: str , dic_types: dict , case_id_column_name: str, object_clean: int, list_cols: list) -> pd.DataFrame: 
    """
    Reads event-log from CSV and process it; lower, removes spaces and '-' from the activity name and others object columns

    Parameters
    -----------------------
    dir_name: str,
        event-log directory
    file_name: str,
        event-log file
    separator: str,
        CSV char separator (',', ';', etc.)
    replace_space: str,
        replace space from string (object) values
    dic_types: dict,
        dictionary of event-lo column types
    case_id_column_name: str,
        name of the case-id column
    object_clean: int,
        if 1, apply lower case and replace characters in string fields (object), except to case-id
    list_cols: list,
        list of columns to be extracted from the log
    Returns
    -----------------------
    Processed event log containing the only the necessary columns for encoding
    """

    path_log = os.path.join(dir_name, file_name)
    df_raw = pd.read_csv(path_log, sep=separator, usecols=list_cols, dtype=dic_types)
    
    if object_clean == 1:
        for column in df_raw.columns:
            if df_raw[column].dtype == 'object' and column!=case_id_column_name:
                df_raw[column] = df_raw[column].str.lower()
                df_raw[column] = df_raw[column].str.replace(' ', replace_space)
                df_raw[column] = df_raw[column].str.replace('-', replace_space)
                df_raw[column] = df_raw[column].str.replace('\'', replace_space)

    return df_raw
