import os
from os import listdir as os_listdir
from os.path import join as os_path_join
from os.path import exists as os_path_exists
from os import makedirs as os_makedirs
from os.path import abspath as os_path_abspath
from os.path import basename as os_path_basename
import pandas as pd
import json

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
    if os_path_exists(path_log) == False:
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

    for file in os_listdir(dir_name):
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

def check_and_create_directory(dir_name:str, dir_parent:str=""):
    """
    Create a directory in its parent directory (optional)

    Parameters
    -----------------------
    dir_name: str,
        directory to be created
    dir_parent: str,
        parent directory in which to create the directory
        
    """

    path_directory = ""
    if dir_parent != "":
        path_directory = os_path_join(dir_parent,dir_name)
    else:
        path_directory = dir_name
    if not os_path_exists(path_directory):
        os_makedirs(path_directory)
        print("The directory '{}' has been created successfully".format(dir_name))
    else:
        print("The directory '{}' already exists".format(dir_name))


def script_info(file):
    """
    Returns information about the script being considered
    """
    script_path = os_path_abspath(file)
    script_name = os_path_basename(script_path)
    return script_path, script_name


def read_json_config(dir_name:str, file_name:str, position:int = 0) -> dict:
    """
    Reads a JSON configuration file and returns the data

    Parameters
    -----------------------
    dir_name: str,
        directory of the JSON file
    file_name: str,
        file name of the JSON file
    position: int,
        the specific position in the "POSITIONS" array to extract

    Returns
    -----------------------
    A dictionary containing the data from the JSON file
    """
    path_data = os_path_join(dir_name, file_name)
    try:
        with open(path_data, 'r') as file:
            config_dict = json.load(file)
            # Extracting data for the specified position
            if "POSITIONS" in config_dict and len(config_dict["POSITIONS"]) > position:
                return config_dict["POSITIONS"][position]
            else:
                print("Position {} not found or out of range".format(position))
                return None
    except FileNotFoundError:
        print("File JSON not found:", path_data)
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON file:", path_data)
        return None
    except Exception:
        print("Generic error on JSON file:", path_data)
        return None
