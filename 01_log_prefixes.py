# 02_log_prefixes
# 1) read a CSV log
# 2) convert the CSV in XES
# 3) extract the prefixes of various lengths from prefix_list
# 4) save stats about the prefix

### IMPORT ###
import os
import csv
from datetime import datetime
import pandas as pd
import pm4py

### LOCAL IMPORT ###
from config.config_reader import ConfigReadYaml

### GLOBALS ###

yaml_config = ConfigReadYaml()

dir_log_in  = yaml_config['DIR_LOG_INPUT']          # Directory with logs from which to derive prefixes 
dir_log_prefix_out = yaml_config['DIR_LOG_PREFIX']  # Directory with prefixes obtained
dir_log_stats = yaml_config['DIR_LOG_STATS'] 
file_prefix_stats = yaml_config['FILE_PREFIX_STATS'] 

log_case_id_col_name = yaml_config['LOG_CASE_ID_NAME']
log_event_col_name = yaml_config['LOG_EVENT_NAME']
log_event_timestamp_col_name = yaml_config['LOG_EVENT_TIMESTAMP_NAME']
log_region_col_name = yaml_config['LOG_REGION_NAME']
log_amount_col_name = yaml_config['LOG_AMOUNT_NAME']
log_tender_type_col_name = yaml_config['LOG_TENDER_TYPE_NAME']
log_label_col_name = yaml_config['LOG_TENDER_DURATION_NAME'] # label / outcome = remaining time
log_duration_ms_col_name = yaml_config['LOG_TENDER_DURATION_MS_NAME']
log_duration_partial_col_name = yaml_config['LOG_TENDER_DURATION_PARTIAL_NAME']
log_sector_col_name = yaml_config['LOG_TENDER_SECTOR_NAME']
log_framework_col_name = yaml_config['LOG_TENDER_FRAMEWORK_NAME']
log_cpv_col_name = yaml_config['LOG_TENDER_CPV_NAME']

prefix_min_num = int(yaml_config['PREFIXES_MIN_NUM'])
prefix_max_num = int(yaml_config['PREFIXES_MAX_NUM'])
prefix_step_num = int(yaml_config['PREFIXES_STEP_NUM'])

# CUSTOMIZE

# prefix dimensions 
prefix_list = list(range(prefix_min_num, prefix_max_num, prefix_step_num)) 

csv_separator = str(yaml_config['CSV_DEFAULT_SEPARATOR'])

### FUNCTIONS ###

def log_global_stats(file_name_prefix:str, df:pd.DataFrame):
    """
    Given the prefix dataframe, calculate time statistics and save it to CSV

    Parameters
    file_name_prefix:str,
        file name with prefixes
    df: pd.DataFrame, 
        dataframe prefix dataframe
    """
    global file_prefix_stats
    # Global (file name, cases, average duration_dd, average duration_partial_dd)
    cases_n = df[log_case_id_col_name].nunique()
    print(df[log_label_col_name].unique())
    duration_dd_avg = df[log_label_col_name].median()
    duration_partial_dd_avg = df[log_duration_partial_col_name].mean()
    duration_dd_med = df[log_label_col_name].mean()
    duration_partial_dd_med = df[log_duration_partial_col_name].median()    
    string_d = "".join([file_name_prefix, ";" , str(cases_n), ";", str(duration_dd_avg), ";", str(duration_dd_med), ";", str(duration_partial_dd_avg),  ";", str(duration_partial_dd_med), os.linesep])
    path_stats = os.path.join(dir_log_stats, file_prefix_stats)
    print("- Saving event log global stats (CSV):", path_prefix)
    with open(path_stats, "a") as fp:
        fp.write(string_d)

def log_global_stats_init():
    """
    Initialise the prefix statistics file
    """
    global file_prefix_stats
    path_stats = os.path.join(dir_log_stats, file_prefix_stats)
    string_h = "".join(["file_name;cases;duration_dd_avg;duration_dd_median;duration_partial_dd_avg;duration_partial_dd_median",os.linesep])
    with open(path_stats, "w") as fp:
        fp.write(string_h)
        
def log_global_stats_xlsx():
    """
    Given the statistics file in CSV, saves it in XLSX
    """
    global file_prefix_stats
    path_stats_csv =  os.path.join(dir_log_stats, file_prefix_stats)
    df_stats = pd.read_csv(path_stats_csv, sep = csv_separator)
    file_stats_xlsx = "".join([file_prefix_stats.split(".")[0],".xlsx"])
    path_stats_xlsx = os.path.join(dir_log_stats, file_stats_xlsx)
    print("- Saving event log global stats (XLSX):", path_stats_xlsx)
    df_stats.to_excel(path_stats_xlsx, index=False)

### MAIN ###

start_time = datetime.now().replace(microsecond=0)

print()
print("*** PROGRAM START ***")
print()

print("Starting time:", start_time)
print()

print(">> Configuration")
print()

print("Prefix list values:", str(prefix_list))
print()

list_files_csv = []

for file in os.listdir(dir_log_in):
    if file.endswith(".csv"):
        list_files_csv.append(file)

list_files_csv_len = len(list_files_csv)

print("Files CSV found (number):", list_files_csv_len)
print("Files CSV found:", list_files_csv)
print()
print()

print(">> Reading CSV files")
print()

if list_files_csv_len == 0:
    print("No CSV files to parse")
else:
    i = 0
    log_global_stats_init()
    for file_csv in list_files_csv:
        i+=1
        print("[{}]".format(i))
        
        # CVS file name
        parts = file_csv.split(".")

        # reading CSV log
        path_log = os.path.join(dir_log_in, file_csv)
        print("Reading:", path_log)
        dic_t = {log_case_id_col_name:object, log_event_col_name:object, log_event_timestamp_col_name:object, log_amount_col_name:float, log_tender_type_col_name:object, log_region_col_name:object, log_sector_col_name:object, log_framework_col_name:object, log_cpv_col_name:int, log_label_col_name:float} # column types
        log_df = pd.read_csv(path_log, sep=csv_separator, dtype=dic_t ,low_memory=False)
        print(log_df.columns) # debug
        ncases = log_df[log_case_id_col_name].nunique()
        nevents = len(log_df)
        print("Number of cases in log:", ncases)
        print("Number of events in log:", nevents)


        # prefixes creation
        print("Creating XES event log")
        dataframe = pm4py.format_dataframe(log_df, case_id=log_case_id_col_name, activity_key=log_event_col_name, timestamp_key=log_event_timestamp_col_name)
        log_df_xes = pm4py.convert_to_event_log(dataframe)
        print("Extracting prefixes...")
        for prefix_len in prefix_list:
            print("Prefix length:", prefix_len)
            trimmed_df_xes = pm4py.get_prefixes_from_log(log_df_xes, length=prefix_len, case_id_key=log_case_id_col_name) # length -> length of every trace
            trimmed_df_csv = pm4py.convert_to_dataframe(trimmed_df_xes)
            ncases = trimmed_df_csv[log_case_id_col_name].nunique()
            print("Number of cases in prefix:", ncases)
            file_name_prefix = "".join([parts[0] , "_" , "P" , "_" , str(prefix_len) , ".csv"])
            path_prefix = os.path.join(dir_log_prefix_out, file_name_prefix)
            print("- Saving log prefix to:", path_prefix)
            trimmed_df_csv.to_csv(path_prefix, sep = ";", index = False, header = True, quoting=csv.QUOTE_ALL)
            print(trimmed_df_csv.head())
            print()
            # Global stats about the trimmed_df_csv
            log_global_stats(file_name_prefix, trimmed_df_csv)

    # At the end of the loop, converts prefix statistics from CSV to XLSX
    log_global_stats_xlsx()

end_time = datetime.now().replace(microsecond=0)

print("End process:", end_time)
print()
print("Time to finish:", end_time - start_time)
print()

print()
print("*** PROGRAM END ***")
print()
