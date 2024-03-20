import pandas as pd

def extract_log_corpus(df: pd.DataFrame, case_id_column_name: str, event_column_name: str, label_column_name: str):
    """
    Extracts the event-log body: case-id, traces, relative labels  

    Parameters
    -----------------------
    df: pd.DataFrame,
        Dataframe containing the event log
    case_id_column_name: str,
        name of the case-id column
    event_column_name: str,
        name of the activity column
    label_column_name: str
        name of the label column

    Returns
    -----------------------
    case_ids: list,
        List of case ids
    traces: list,
        List of traces
    y: list,
        List of labels
    """

    # Empty lists
    traces, y, case_ids = [], [], []
    
    for group in df.groupby(case_id_column_name):
        events = list(group[1][event_column_name])
        traces.append(' '.join(x for x in events))
        y.append(list(group[1][label_column_name])[0])
        case_ids.append(list(group[1][case_id_column_name])[0])

    return case_ids, traces, y