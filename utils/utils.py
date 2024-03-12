import os
from os.path import join as os_path_join
from os.path import exists as os_path_exists
from os import makedirs as os_makedirs
import datetime 
import pandas as pd

def seconds_to_hours(seconds:int) -> str:
    """
    Convert seconds into hours, minutes and seconds as a string

    Parameters
    -----------------------
    seconds: int,
        value in seconds

    Returns
    -----------------------
    String in hours:minutes:seconds
    """

    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hours, minutes, seconds)


def app_log_init(dir_name:str, file_name:str, app_log_file_header:list):
    """
    Initialise the app log to trace the timing

    Parameters
    -----------------------
    dir_name: str,
        directory name of the app-log
    file_name: str,
        file name of the app-log
    """

    string_csv = ";".join(app_log_file_header)
    path_log = os_path_join(dir_name,file_name)
    if os.path.exists(path_log) == False:
        with open(path_log,"w") as fp:
            fp.write(string_csv)
            fp.write(os.linesep)


def app_log_write(dir_name:str, file_name:str, app_log_dic:dict):
    """
    Write on the app log to trace the overall timing of the script
    """

    string_csv = "".join([app_log_dic["script_name"], ";", app_log_dic["model"], ";", app_log_dic["configuration"], ";", str(app_log_dic["start_time"]), ";", str(app_log_dic["end_time"]), ";", str(app_log_dic["delta_time"])])
    path_log = os_path_join(dir_name,file_name)
    with open(path_log,"a") as fp:
        fp.write(string_csv)
        fp.write(os.linesep)


def file_list_by_type(dir_name, file_type):
    """
    Creates the list of files in a directory (dir_name) of a specific file_type
    """
    
    list_files = []
    
    file_ext = "".join([".", file_type])

    for file in os.listdir(dir_name):
        if file.endswith(file_ext):
            list_files.append(file)

    return list_files


def get_llm_data(path_llm:str, csv_separator:str) -> pd.DataFrame:
    """
    Extracts data obtained from the LLM
    """
    dic_llm = {"case_id":object, "tender_increase_label":float}
    col_llm = ["case_id", "tender_increase_label"]
    df_llm = pd.read_csv(path_llm, sep=csv_separator, dtype=dic_llm, usecols=col_llm, low_memory=False)
    return df_llm

def check_and_create_directory(directory:str, parent_dir:str=""):
    """
    Create a directory in its parent directory (optional)

    Parameters
    -----------------------
    directory: str,
        directory to be created
    parent_dir: str,
        parent directory in which to create the directory
        
    """

    path_directory = ""
    if parent_dir != "":
        path_directory = os_path_join(parent_dir,directory)
    else:
        path_directory = directory
    if not os_path_exists(path_directory):
        os_makedirs(path_directory)
        print("The directory '{}' has been created successfully".format(directory))
    else:
        print("The directory '{}' already exists".format(directory))