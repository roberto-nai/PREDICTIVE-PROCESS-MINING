# 00_log_disco_duration.py

### IMPORT ###
import os
from datetime import datetime
import csv
import numpy as np
import pandas as pd

### LOCAL IMPORT ###
from config.config_reader import ConfigReadYaml

### GLOBALS ###

yaml_config = ConfigReadYaml()
dir_log_disco = yaml_config['DIR_LOG_DISCO'] 
dir_log_disco_duration = yaml_config['DIR_LOG_DISCO_DURATION'] 
dir_log = yaml_config['DIR_LOG_INPUT'] 
dir_log_stats = yaml_config['DIR_LOG_STATS']  
dir_log_exclude = yaml_config['DIR_EXCLUDE']
file_log_exclude = yaml_config['FILE_EXCLUDE']

list_case_id_exc = [] # list of case ids to be excluded from the event-log

log_case_id_col_name = yaml_config['LOG_CASE_ID_NAME']
log_event_col_name = yaml_config['LOG_EVENT_NAME']
log_event_timestamp_col_name = yaml_config['LOG_EVENT_TIMESTAMP_NAME']
log_region_col_name = yaml_config['LOG_REGION_NAME']
log_amount_col_name = yaml_config['LOG_AMOUNT_NAME']
log_tender_type_col_name = yaml_config['LOG_TENDER_TYPE_NAME']
log_duration_col_name = yaml_config['LOG_TENDER_DURATION_NAME']
log_duration_ms_col_name = yaml_config['LOG_TENDER_DURATION_MS_NAME']
log_duration_partial_col_name = yaml_config['LOG_TENDER_DURATION_PARTIAL_NAME']
log_sector_col_name = yaml_config['LOG_TENDER_SECTOR_NAME']
log_framework_col_name = yaml_config['LOG_TENDER_FRAMEWORK_NAME']
log_cpv_col_name = yaml_config['LOG_TENDER_CPV_NAME']


log_case_id_len_col_name = yaml_config['LOG_CASE_ID_LEN']

csv_separator = str(yaml_config['CSV_DEFAULT_SEPARATOR'])

log_disco_duration = "Cases in log_2016-2022_clean.csv" # <-- INPUT (cases from DISCO)

# dic_t = {"case_id":object,"date":object,"event":object,"lots":int,"amount":float,"type":object,"sector":object, "region":object,"case_id_fre":int,"complaint":int, "duration_ms":float} # types

### FUNCTIONS ###
def list_cases_exclude() -> list:
    """Gets the list of cases to be excluded from the event-log"""
    string_cases = ""
    path_file = os.path.join(dir_log_exclude, file_log_exclude)
    
    with open(path_file, "r") as fp:
        string_cases = fp.read()
    
    list_string = string_cases.split(",")

    list_case_id_exc = [str(elemnt) for elemnt in list_string]

    return list_case_id_exc

### MAIN ###

print()
print("*** PROGRAM START ***")
print()

start_time = datetime.now().replace(microsecond=0)

print("Starting time:", start_time)
print()

print(">> Configuration")
print()

list_case_id_exc = list_cases_exclude()

print("Input directory with DISCO logs in CSV:", dir_log_disco)
print("Output directory with logs in CSV with duration:", dir_log)
print("Cases to be excluded from event-log:", list_case_id_exc)

list_files_csv = []

for file in os.listdir(dir_log_disco):
    if file.endswith(".csv"):
        list_files_csv.append(file)

list_files_csv_len = len(list_files_csv)

print("Files CSV found (number):", list_files_csv_len)
print("Files CSV found:", list_files_csv)
print()
print()

if list_files_csv_len == 0:
    print("No CSV files to parse")
else:
    i = 0
    print(">> Reading CSV files")
    for file_csv in list_files_csv:
        i+=1
        print("[{} / {}]".format(i, list_files_csv_len))
        
        # CSV file name
        parts = file_csv.split(".")

        # Read the main log
        path_log_disco = os.path.join(dir_log_disco, file_csv)
        print("Reading:", path_log_disco)
        # col_sel = [] # all the columns
        # col_sel = ['Case ID','Activity','Resource','Complete Timestamp','amount','type'] # columns to be read
        col_sel = ['Case ID','Activity','Resource','Complete Timestamp','amount','type', 'sector', 'framework', 'cpv'] # columns to be read
        dic_t = {"Case ID":object, "Activity":object, "Resource":object, "Complete Timestamp":object, "amount":float, "type":object, "sector":object, "framework":int, "cpv":int} # column types
        if len(col_sel) > 0:
            df_log = pd.read_csv(path_log_disco, sep=',', usecols=col_sel, dtype=dic_t ,low_memory=False)
        else:
            df_log = pd.read_csv(path_log_disco, sep=',', dtype=dic_t ,low_memory=False)

        # Rename DISCO columns with those defined in config.yml
        df_log.rename(columns={"Case ID": log_case_id_col_name, "Activity": log_event_col_name, "Complete Timestamp": log_event_timestamp_col_name, "Resource":log_region_col_name, "amount": log_amount_col_name, "type": log_tender_type_col_name, "sector": log_sector_col_name, "framework": log_framework_col_name, "cpv":log_cpv_col_name}, inplace=True)

        # Exclude from the log the cases in list_case_id_exc
        df_log = df_log[~df_log[log_case_id_col_name].isin(list_case_id_exc)]

        # Rearrange columns (INPUT)
        df_log = df_log[[log_case_id_col_name, log_event_col_name, log_event_timestamp_col_name, log_region_col_name, log_amount_col_name, log_tender_type_col_name, log_sector_col_name, log_framework_col_name, log_cpv_col_name]]

        print("LOG columns:")
        print(df_log.columns)
        print("LOG shape:")
        print(df_log.shape)
            
        num_events = len(df_log)
        num_cases = df_log['case_id'].nunique()
        print("Main log informations")
        print("Number of cases: {}".format(num_cases))
        print("Number of events: {}".format(num_events))
        print()

        # Add the log duration and events number (from DISCO)
        path_log_disco = os.path.join(dir_log_disco_duration, log_disco_duration)
        print("Reading:", path_log_disco)
        col_sel = ['Case ID', 'Events', 'Duration (milliseconds)'] # columns to be read from DISCO
        dic_t = {"Case ID":object, "Duration (milliseconds)":int} # column types
        df_log_duration = pd.read_csv(path_log_disco, sep=',', usecols=col_sel, dtype=dic_t ,low_memory=False)
        df_log_duration.rename(columns={"Case ID": log_case_id_col_name, "Duration (milliseconds)": log_duration_ms_col_name}, inplace=True)

        # Exclude from the log the cases in list_case_id_exc
        df_log_duration = df_log_duration[~df_log_duration[log_case_id_col_name].isin(list_case_id_exc)]

        print("LOG duration columns:")
        print(df_log_duration.columns)

        # Join
        df_log_final = pd.merge(left=df_log, right=df_log_duration, on=log_case_id_col_name, how = 'left')

        # ms to days
        # days = round((millis/86400000), 2)
        df_log_final[log_duration_col_name] = round((df_log_final[log_duration_ms_col_name]/86400000), 2)
        # drop duration_ms
        df_log_final = df_log_final.drop(columns=[log_duration_ms_col_name])

        # df_log_final[log_event_timestamp_col_name] = df_log_final[log_event_timestamp_col_name].str.replace(' ', replace_space)
        # In the final log, it adds the partial durations (in days)
        # 'data_temp': log_event_timestamp_col_name in datetime
        # 'date_diff': time elapsed since the previous event
        df_log_final['data_temp'] = pd.to_datetime(df_log_final[log_event_timestamp_col_name])
        df_log_final['date_diff'] = df_log_final.groupby(log_case_id_col_name)['data_temp'].diff() / np.timedelta64(1, 'D') 
        df_log_final['date_diff'] = df_log_final['date_diff'].fillna(0)
        df_log_final['date_diff'] = df_log_final['date_diff'].astype(int)
        df_log_final[log_duration_partial_col_name] = 0
        df_log_final[log_duration_partial_col_name] = df_log_final.groupby(log_case_id_col_name)['date_diff'].cumsum()
        # Removes temporary columns
        df_log_final = df_log_final.drop(columns=['data_temp', 'date_diff'])

        # Stats about the traces in the event-log (case_id, number of events, total duration), using the 'Events' to get the number of events in a trace
        df_log_final_stats = df_log_final[[log_case_id_col_name, 'Events', log_duration_col_name]]
        df_log_final_stats = df_log_final_stats.rename(columns={'Events': log_case_id_len_col_name})
        df_log_final_stats = df_log_final_stats.drop_duplicates(subset=[log_case_id_col_name])
        df_log_final_stats = df_log_final_stats.sort_values(by=[log_duration_col_name, log_case_id_len_col_name, log_case_id_col_name], ascending=True)

        # drop 'Events' column
        df_log_final = df_log_final.drop(columns=['Events'])

        print("LOG final shape:")
        print(df_log_final.shape)
        print("LOG final columns:")
        print(df_log_final.columns)
        print()

        # Save df_log_final in CSV
        print(">> Saving")
        path_log_disco = os.path.join(dir_log, file_csv)
        df_log_final.to_csv(path_log_disco, sep = csv_separator, index=False, quoting=csv.QUOTE_ALL)
        print("Path event-log:", path_log_disco)
        
        # Save df_log_final_stats in CSV ans XLSX
        file_csv_stats = "".join([parts[0], "_", "stats", ".csv"])
        path_log_stats = os.path.join(dir_log_stats, file_csv_stats)
        print("Path event-log stats:", path_log_stats)
        print()
        # sheet_name must be <= 31 chars
        parts_sheet_name = parts[0].split("_")
        sheet_name = parts_sheet_name[-1]
        df_log_final_stats.to_csv(path_log_stats, sep = csv_separator, index=False, quoting=csv.QUOTE_ALL)
        file_xlsx_stats = "".join([parts[0], "_", "stats", ".xlsx"])
        path_log_stats = os.path.join(dir_log_stats, file_xlsx_stats)
        df_log_final_stats.to_excel(path_log_stats, index=False, sheet_name=sheet_name)
        print()

end_time = datetime.now().replace(microsecond=0)

print("End process:", end_time)
print()
print("Time to finish:", end_time - start_time)
print()

print()
print("*** PROGRAM END ***")
print()
