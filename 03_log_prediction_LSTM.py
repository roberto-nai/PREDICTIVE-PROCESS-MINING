### IMPORT ###
import os
import math 
import pandas as pd
from datetime import datetime
from os.path import abspath as path_abspath
from os.path import basename as path_basename
from os.path import join as path_join
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout

### LOCAL IMPORT ###
from config.config_reader import ConfigReadYaml
from utils.utils import app_log_init, app_log_write, file_list_by_type, get_llm_data, check_and_create_directory

### GLOBALS ###

yaml_config = ConfigReadYaml()
dir_log_encoded = yaml_config['DIR_LOG_ENCODED'] # data input of the model
dir_models_results = yaml_config['DIR_MODELS_RESULTS'] 
csv_separator = str(yaml_config['CSV_DEFAULT_SEPARATOR'])

# LOG features
log_amount_col_name = yaml_config['LOG_AMOUNT_NAME']

# LLM
llm_do = int(yaml_config['LLM_DO']) # yes / no
dir_log_llm = yaml_config['DIR_LOG_LLM'] 
log_llm_file = yaml_config['FILE_LOG_LLM'] 
path_llm = os.path.join(dir_log_llm, log_llm_file)

# APP-LOG (SCRIPT)
app_log_dir = yaml_config['DIR_APP_LOG']
app_log_file = yaml_config['FILE_APP_LOG']
script_path = path_abspath(__file__)
script_name = path_basename(script_path) # file name of this script
app_log_file_header = list(yaml_config['APP_LOG_HEADER'])
app_log_dic = dict(yaml_config['APP_LOG_DIC'])

# LSTM settings
model_epochs = int(yaml_config['EPOCHS_NUM']) # 200
optimizer_name = str(yaml_config['OPTIMIZER_NAME']) # "adam"
loss_name = str(yaml_config['LOSS_NAME'])  # "mse"

### INPUT ###
model_suffix = "LSTM" # <-- INPUT: enter the desired model prefix
# list_col_exclude = [log_amount_col_name] # <-- INPUT: enter the columns to be excluded from the data, else []
list_col_exclude = [] # <-- INPUT: enter the columns to be excluded from the data, else []
list_col_exclude_len = len(list_col_exclude)

### OUTPUT ###
string_llm = "".join(["LLM-",str(llm_do)])
string_col_exclude = "".join(["FEATEXC-",str(list_col_exclude_len)])
results_configuration = "".join(["regression_", model_suffix, "_", string_col_exclude, "_" , string_llm]) # configuration
file_results_csv = "".join([results_configuration, ".csv"]) # CSV file output
file_results_xlsx = "".join([results_configuration, ".xlsx"]) # XLSX file output

### MAIN ###

print()
print("*** PROGRAM START ***")
print()

start_time = datetime.now().replace(microsecond=0)

# app-log init
app_log_init(app_log_dir, app_log_file, app_log_file_header)

# Creation of output directories
print(">> Creating output directories")
print()
check_and_create_directory(app_log_dir)
check_and_create_directory(dir_models_results)
print()

print(">> Configuration")
print()
print("LLM do:", llm_do)
print("Model suffix name:", model_suffix)
print()

print(">> Listing CSV files")
print()

list_files_csv = []

list_files_csv = file_list_by_type(dir_log_encoded, "csv")
list_files_csv_len = len(list_files_csv)

print("Files CSV found (number):", list_files_csv_len)
print("Files CSV found:", list_files_csv)
print()

if list_files_csv_len == 0:
    print("No CSV files to parse")
else:
    i = 0

    print(">> Reading CSV files")
    print()

    list_dic_res = [] # list with result dictionaries

    for file_csv in list_files_csv:
        i+=1
        print("[{} / {}]".format(i, list_files_csv_len))
        print()
        
        # CVS file name
        parts = file_csv.split(".")

        # CSV metadata (prefix length and encoding)
        metadata = parts[0].split("_")
        prefix_len = metadata[-2]
        prefix_enc = metadata[-1]

        # Load the dataset (the log encoded)
        path_csv = path_join(dir_log_encoded, file_csv)
        print("File name:", file_csv)
        print("File path:", path_csv)
        print()

        df_log = pd.read_csv(path_csv, delimiter=csv_separator)
        print("Data preview:")
        print(df_log.head())
        print()

        # Removes features (if needed)
        print(">> Removing columns")
        print()
        print("Columns to remove:", list_col_exclude_len)
        print()
        if list_col_exclude_len > 0:
            print("Features before removing ({}): {}".format(len(df_log.columns),df_log.columns))
            for column_name in list_col_exclude:
                if column_name in df_log.columns:
                    print("Removing:", column_name)
                    df_log = df_log.drop(columns=[column_name])
                    print("Features after removing ({}): {}".format(len(df_log.columns),df_log.columns))
                    print()

        # Add the LLM extracted feature
        if llm_do == 1:
            print(">> Getting LLM data:", path_llm)
            print()
            df_llm = get_llm_data(path_llm, csv_separator)
            print("Features before LLM merge:", df_log.columns)
            print()
            # print(df_llm.head()) # debug
            df_log = pd.merge(df_log, df_llm, on="case_id", how="left")
            # df_log[log_amount_col_name] = df_log[log_amount_col_name] + df_log["tender_increase_label"]
            # df_log = df_log.drop("tender_increase_label", axis=1)
            # empty_increase = df_log[df_log["tender_increase_label"].isna()]
            df_log["tender_increase_label"] = df_log["tender_increase_label"].fillna(0)
            print("Dataframe after LLM data merge")
            print(df_log.head()) # debug: data after LLM append
            print()
            print("Features after LLM merge:", df_log.columns)
            print()

        # Remove case_id
        df_log = df_log.drop(columns=['case_id'])

        # Split features (X) and labels (y)
        X = df_log.drop(columns=['label'])
        y = df_log['label'].values

        # Number of features
        features_num = len(X.columns)
        # print("Number of features (excluding ID and LABEL)", features_num) # debug

        ## LSTM (begin)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize the data using the MinMax tecnique
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Reshape data for LSTM
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Create the LSTM model
        model = Sequential()
        model.add(LSTM(units = model_epochs, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2)) # Avoid overfitting

        model.add(LSTM(units = model_epochs, return_sequences = True))
        model.add(Dropout(0.2)) # Avoid overfitting

        model.add(LSTM(units = model_epochs))
        model.add(Dropout(0.2)) # Avoid overfitting

        model.add(Dense(1))

        model.summary() # displays the model configuration

        # Compile the model
        # loss_name = "mse"
        # optimizer_name = "adam"
        model.compile(optimizer=optimizer_name, loss=loss_name)

        # Fit the model
        model.fit(X_train, y_train, epochs=model_epochs, batch_size=32, verbose=1)

        # Evaluate the model
        loss = model.evaluate(X_test, y_test)
        print('Test Loss ({}): {}'.format(loss_name,loss))

        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # RMSE
        rmse_train = math.sqrt(mean_squared_error(y_train, train_predict))
        rmse_test = math.sqrt(mean_squared_error(y_test, test_predict))
        print("RMSE test data on dataset '{}':".format(file_csv,rmse_test))
        print()

        ## LSTM (end)

        # get min and max values from y_test (the duration labels)
        min_test = min(y_test)
        max_test = max(y_test)
        rmse_test_norm = rmse_test / (max_test - min_test) # normalized RMSE
        print("RMSE normalized data on dataset '{}':".format(file_csv,rmse_test_norm))

        # save the results in a dictionary
        dic_res = {'file_name': file_csv, 'prefix_length':prefix_len, 'prefix_encoding':prefix_enc, 'model': model_suffix, 'features_num':features_num, 'features_excluded': list_col_exclude_len, 'LLM_data':llm_do, 'epochs': model_epochs, 'optimizer_name':optimizer_name, 'loss':loss_name, 'RMSE_model': rmse_test, 'RMSE_norm':rmse_test_norm}
        list_dic_res.append(dic_res)

print()

print(">> Saving {} results".format(model_suffix))
print()

# Create the dataframe with results
df_results = pd.DataFrame.from_records(list_dic_res)
df_results = df_results.sort_values(by = 'RMSE_norm', ascending=True)
# print(df_results.head()) # debug

# Save results in CSV
path_results = path_join(dir_models_results, file_results_csv)
print("Saving results to:", file_results_csv)
df_results.to_csv(path_results, sep = csv_separator, index=False)
print()
# Save results in XLSX
path_results = path_join(dir_models_results, file_results_xlsx)
print("Saving results to :", file_results_xlsx)
df_results.to_excel(path_results, index=False, sheet_name=model_suffix)
print()

end_time = datetime.now().replace(microsecond=0)
delta_time = end_time - start_time

print("End process:", end_time)
print()
print("Time to finish:", delta_time)
print()

# APP-LOG tracing
app_log_dic["script_name"] = script_name
app_log_dic["model"] = model_suffix
app_log_dic["configuration"] = results_configuration
app_log_dic["start_time"] = start_time
app_log_dic["end_time"] = end_time
app_log_dic["delta_time"] = delta_time
app_log_write(app_log_dir, app_log_file, app_log_dic)

print()
print("*** PROGRAM END ***")
print()

