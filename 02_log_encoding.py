# 02_log_encoding.py
# Prefixes encoding

### IMPORT ###
import os
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

### LOCAL IMPORT ###
from config.config_reader import ConfigReadYaml
from utils import read_log
from utils import extract_log_corpus
from utils import extract_log_attribute
from utils import encoding_simple_index
from utils import encoding_binary
from utils import encoding_frequency

### GLOBALS ###

yaml_config = ConfigReadYaml()

dir_log_prefix  = yaml_config['DIR_LOG_PREFIX'] 
dir_log_encoded = yaml_config['DIR_LOG_ENCODED'] 

csv_separator = str(yaml_config['CSV_DEFAULT_SEPARATOR'])
replace_char = "_" # To replace spaces or '-'

# CUSTOMIZE with log names
log_case_id_col_name = yaml_config['LOG_CASE_ID_NAME']
log_event_col_name = yaml_config['LOG_EVENT_NAME']
log_event_timestamp_col_name = yaml_config['LOG_EVENT_TIMESTAMP_NAME']
log_region_col_name = yaml_config['LOG_REGION_NAME']
log_amount_col_name = yaml_config['LOG_AMOUNT_NAME']
log_tender_type_col_name = yaml_config['LOG_TENDER_TYPE_NAME']
log_duration_col_name = yaml_config['LOG_TENDER_DURATION_NAME'] # label / outcome = remaining time
log_label_col_name = yaml_config['LOG_LABEL_NAME'] # label column name
log_duration_partial_col_name = yaml_config['LOG_TENDER_DURATION_PARTIAL_NAME']
log_sector_col_name = yaml_config['LOG_TENDER_SECTOR_NAME']
log_framework_col_name = yaml_config['LOG_TENDER_FRAMEWORK_NAME']
log_cpv_col_name = yaml_config['LOG_TENDER_CPV_NAME']

encoding_suffix_dic = {'I': "Simpler-Index", 'B': "Binary", 'F': "Frequency"}

# encoding_suffix = "I" # <-- INPUT: B = Binary, F = Frequency, I = simple-index

encoding_suffix_list = ["B", "F", "I"] # <-- INPUT: B = Binary, F = Frequency, I = simple-index
encoding_suffix_list_len = len(encoding_suffix_list)

### MAIN ###

start_time = datetime.now().replace(microsecond=0)

print()
print("*** PROGRAM START ***")
print()

print("Starting time:", start_time)
print()

print(">> Configuration")
print()

print("Input directory:", dir_log_prefix)
print("Output directory:", dir_log_encoded)
print()

list_files_csv = []

for file in os.listdir(dir_log_prefix):
    if file.endswith(".csv"):
        list_files_csv.append(file)

list_files_csv_len = len(list_files_csv)

print("Files CSV found (number):", list_files_csv_len)
print("Files CSV found:", list_files_csv)
print("Encoding types ({}): {}".format(encoding_suffix_list_len, encoding_suffix_list))
print()

print(">> Reading CSV files")
print()

if list_files_csv_len == 0:
    print("No CSV files to parse")
else:
    i = 0
    total_steps = list_files_csv_len * encoding_suffix_list_len
    for encoding_suffix in encoding_suffix_list:
        print("Encoding:", encoding_suffix_dic[encoding_suffix])
        for file_csv in list_files_csv:
            i+=1
            print("[{} / {}]".format(i, total_steps))
            print()
            
            # CVS file name
            parts = file_csv.split(".")
            # prefix_len = parts[-2]

            # Reading CSV
            print(">> Reading:", file_csv)
            print()

            # Options
            clean_log = 1
            # Set the columns wanted from the event-log and the relative types
            list_colums = [log_case_id_col_name, log_event_col_name, log_event_timestamp_col_name, log_amount_col_name, log_tender_type_col_name, log_region_col_name, log_duration_partial_col_name, log_duration_col_name, log_sector_col_name, log_framework_col_name, log_cpv_col_name] # columns wanted
            dic_t = {log_case_id_col_name:object, log_event_col_name:object, log_event_timestamp_col_name:object, log_amount_col_name:float, log_tender_type_col_name:object, log_region_col_name:object, log_duration_partial_col_name: int, log_duration_col_name:object, log_sector_col_name:object, log_framework_col_name:int, log_cpv_col_name:object} # column types
            
            # Gets the event-log in DF
            log_df = read_log(dir_log_prefix, file_csv, csv_separator, replace_char, dic_t, log_case_id_col_name, clean_log, list_colums)
            
            print("Input event-log columns:")
            print(log_df.columns)
            print()
            ncases = log_df[log_case_id_col_name].nunique()
            nevents = len(log_df)
            print("Number of cases in log:", ncases)
            print("Number of events in log:", nevents)
            # conteggio_per_id = log_df.groupby(log_case_id_col_name).size().reset_index(name='count_unique')
            print()

            # encoding the trace
            print(">> Encoding...")
            print()
            
            # ENCODING LOGIC

            # read event-log and export case-id, activities and labels (y) in separate columns (lists)
            ids, traces, y = extract_log_corpus(log_df, log_case_id_col_name, log_event_col_name, log_duration_col_name)

            out_df = None

            if encoding_suffix == 'I':
                out_df = encoding_simple_index(traces)

            if encoding_suffix == 'B':
                out_df = encoding_binary(traces)
            
            if encoding_suffix == 'F':
                out_df = encoding_frequency(traces)

            print("Encoded DF shape:", out_df.shape) # debug
            print()

            # /ENCODING LOGIC

            # After the specific encoding, extract the trace attributes (and encode them if necessary)
            amounts = extract_log_attribute(log_df, log_case_id_col_name, log_amount_col_name)
            # types = extract_log_attribute(log_df, log_case_id_col_name, log_tender_type_col_name) # type is not necessary, as they are all of the same type grouped by file
            regions = extract_log_attribute(log_df, log_case_id_col_name, log_region_col_name) # to be encoded
            sectors = extract_log_attribute(log_df, log_case_id_col_name, log_sector_col_name) # to be encoded
            frameworks = extract_log_attribute(log_df, log_case_id_col_name, log_framework_col_name) 
            cpvs = extract_log_attribute(log_df, log_case_id_col_name, log_cpv_col_name) # to be encoded

            # partial duration: the last (row) partial duration of the prefix
            attribute_partial_duration = []
            df_last_row_by_id = log_df[~log_df.duplicated(log_case_id_col_name, keep='last')]
            partial_dd = list(df_last_row_by_id[log_duration_partial_col_name])

            # add the vectors (attributes) to the matrix
            out_df[log_amount_col_name] = amounts
            # out_df['tender_type'] = types
            out_df[log_region_col_name] = regions
            out_df[log_sector_col_name] = sectors
            out_df[log_framework_col_name] = frameworks
            out_df[log_cpv_col_name] = cpvs
            
            # out_df = pd.get_dummies(out_df, columns=['tender_type', 'region'], prefix=['type', 'region']) # encode the object attributes
            out_df = pd.get_dummies(out_df, columns=[log_region_col_name, log_sector_col_name, log_cpv_col_name], prefix=['region', 'sector', 'cpv']) # encode the columns object
            # out_df = pd.get_dummies(out_df, columns=[log_region_col_name], prefix=['region']) # encode the object attribute

            # Inserts in first position the case_id
            out_df.insert(loc = 0, column = log_case_id_col_name, value = ids)
            # Inserts in semi-last position partial duration
            out_df.insert(len(out_df.columns), log_duration_partial_col_name, partial_dd)
            # Inserts in last position the label (y)
            out_df.insert(len(out_df.columns), log_label_col_name, y)

            print("Output event-log columns (encoded):")
            print(out_df.columns)
            # print(out_df.head()) # debug
            print()

            print("done!")
            print()

            # Save
            log_file_encoded = parts[0] + "_" + encoding_suffix + ".csv"
            path_encoded = os.path.join(dir_log_encoded, log_file_encoded)
            print(">> Saving:", path_encoded)
            out_df.to_csv(path_encoded, sep = ";", index=False)
            print("done!")
            print()

            print("-"*6)
            print()


end_time = datetime.now().replace(microsecond=0)

print("End process:", end_time)
print()
print("Time to finish:", end_time - start_time)
print()

print()
print("*** PROGRAM END ***")
print()
