import pandas as pd

def extract_log_attribute(df: pd.DataFrame, case_id_column_name: str, log_attribute_name: str):
    """
    Extracts a specific event-log attribute

    Parameters
    -----------------------
    df: pd.DataFrame,
        Dataframe containing the event log
    case_id_column_name: str,
        name of the case-id column
    log_attribute_name: str,
        name of the attribute to be extracted
        
    Returns
    -----------------------
    attribute: list,
        List with attribute values
    """

    # Empty list
    attribute = []
    
    for group in df.groupby(case_id_column_name):
        # print(group[1][log_attribute_name].dtype)
        attribute.append(list(group[1][log_attribute_name])[0])
    return attribute