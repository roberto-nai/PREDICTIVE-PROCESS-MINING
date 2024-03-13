# 03_log_prediction.py

# Random Forest Regression 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# Gradient Boosting Regression
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html

# 2024-01-18: added metric -> Normalized RMSE = RMSE / (max value – min value)
# 2024-01-14: added StandardScaler for amount column (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
# 2024-02-08: added the ability to exclude columns from the dataframe

### IMPORT ###
from datetime import datetime
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # to save SHAP figures
from os.path import join as os_path_join
import sys
# ML models
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy.stats import uniform, randint
import joblib # joblib save the model in binary
# SHAP
import shap

### LOCAL IMPORT ###
from config.config_reader import ConfigReadYaml
from utils.utils import seconds_to_hours, app_log_init, app_log_write, file_list_by_type, get_llm_data, check_and_create_directory, script_info

### GLOBALS ###

yaml_config = ConfigReadYaml()

dir_log_encoded = yaml_config['DIR_LOG_ENCODED'] # data input of the model
dir_models_results = yaml_config['DIR_MODELS_RESULTS'] 
dir_models_dump = yaml_config['DIR_MODELS_DUMP'] 
dir_models_shap = yaml_config['DIR_MODELS_SHAP'] 
dir_models_plot = yaml_config['DIR_MODELS_PLOT'] 

# LLM
llm_do = int(yaml_config['LLM_DO']) # yes / no
dir_log_llm = yaml_config['DIR_LOG_LLM'] 
log_llm_file = yaml_config['FILE_LOG_LLM'] 
path_llm = os_path_join(dir_log_llm, log_llm_file)

csv_separator = str(yaml_config['CSV_DEFAULT_SEPARATOR'])
order_metric = str(yaml_config['COL_ORDER_METRIC'])

# LOG features
log_case_id_col_name = yaml_config['LOG_CASE_ID_NAME']
log_event_col_name = yaml_config['LOG_EVENT_NAME']
log_event_timestamp_col_name = yaml_config['LOG_EVENT_TIMESTAMP_NAME']
log_region_col_name = yaml_config['LOG_REGION_NAME']
log_amount_col_name = yaml_config['LOG_AMOUNT_NAME']
log_tender_type_col_name = yaml_config['LOG_TENDER_TYPE_NAME']
log_label_col_name = yaml_config['LOG_LABEL_NAME'] # label column name
log_sector_col_name = yaml_config['LOG_TENDER_SECTOR_NAME']
log_framework_col_name = yaml_config['LOG_TENDER_FRAMEWORK_NAME']
log_cpv_col_name = yaml_config['LOG_TENDER_CPV_NAME']

cv_folds_num = yaml_config['CV_FOLDS_NUM']
list_col_scale = [] # list of columns to be scaled

# PREFIX SIZES
prefix_min_num = int(yaml_config['PREFIXES_MIN_NUM'])
prefix_max_num = int(yaml_config['PREFIXES_MAX_NUM'])
# prefix_max_num = 6 # <-- INPUT: reduced prefix number for testing 
prefix_step_num = int(yaml_config['PREFIXES_STEP_NUM'])
prefix_list = list(range(prefix_min_num, prefix_max_num, prefix_step_num)) 

# MODELS DUMP
model_dump = int(yaml_config['MODEL_DUMP']) # yes / no

# APP-LOG (SCRIPT)
app_log_dir = yaml_config['DIR_APP_LOG']
app_log_file = yaml_config['FILE_APP_LOG']
script_path, script_name = script_info(__file__)
app_log_file_header = list(yaml_config['APP_LOG_HEADER'])
app_log_dic = dict(yaml_config['APP_LOG_DIC'])

### INPUT ###
model_suffix = "XGR" # <-- INPUT: enter the desired model prefix (see *_PREFIX in config.yml) [RFR, XGR]
# list_col_exclude = [log_amount_col_name] # <-- INPUT: enter the columns to be excluded from the data, else []
list_col_exclude = [] # <-- INPUT: enter the columns to be excluded from the data, else []
list_col_exclude_len = len(list_col_exclude)

### OUTPUT ###
string_llm = "".join(["LLM-",str(llm_do)])
string_col_exclude = "".join(["FEATEXC-",str(list_col_exclude_len)])
results_configuration = "".join(["regression_", model_suffix, "_", string_col_exclude, "_" , string_llm]) # configuration
file_results_csv = "".join([results_configuration, ".csv"]) # CSV file output
file_results_xlsx = "".join([results_configuration, ".xlsx"]) # XLSX file output

### FUNCTIONS ###

def read_log_encoded(dir_name: str, file_name: str, list_col_exclude:list) -> pd.DataFrame:
    """
    Reads file the data for the prediction model

    Parameters
    -----------------------
    dir_name: str,
        directory with file
    file_name: str,
        file name
    list_col_exclude: list,
        list of columns to be excluded

    Returns
    -----------------------
    The dataframe with the data
    """

    path_data = os_path_join(dir_name, file_name)

    df = pd.read_csv(path_data, sep=csv_separator, low_memory=False)

    # Excluding
    for col_name in list_col_exclude:
        if col_name in df.columns: # check if the column from the list is really in the dataframe
            df = df.drop(columns=[col_name])

    return df

def log_encoded_extract_data(df_in: pd.DataFrame, case_id_column_name: str, label_column_name:str, list_col_scale:list):
    """
    Reads file containing the encodings for a given event log and return the matrix of features with relative labels. Scale the amount column before returning.

    Parameters
    -----------------------
    df_in: pd.DataFrame,
        the input dataframe
    case_id_column_name: str,
        name of the case-id column
    label_column_name: str,
        name of the label column
    list_col_scale: list,
        list of columns on which to apply StandardScaler
    
    Returns
    -----------------------
    The encoded vectors (X) as numpy.ndarray, the names of the features in X as list and their corresponding labels (y) as list
    """

    # print(df_in.columns) # debug    
    print("Input dataframe shape:", df_in.shape)
    print()

    # Scaling
    for col_name in list_col_scale:
        if col_name in df_in.columns: # check if the column from the list is really in the dataframe
            print("Scaling", col_name, end="")
            scale_column = df_in[col_name].values.reshape(-1, 1)
            scaler = StandardScaler()
            column_standardized = scaler.fit_transform(scale_column)
            # replace the original column with the standardized values
            # df_in[log_amount_col_name] = column_standardized
            df_in[col_name] = column_standardized
            print("...done!")

    # print(df_in.columns) # debug    
    print("Input DataFrame shape:", df_in.shape)
    print()
        
    # labels (y): list
    y = list(df_in[label_column_name])

    # removes case-ids and labels from the dataframe
    del df_in[case_id_column_name], df_in[label_column_name]
    
    # data matrix (X): numpy.ndarray
    vectors = df_in.to_numpy()
    vectors_features = df_in.columns.to_list()
    
    del df_in

    # print(type(vectors)) # numpy.ndarray
    # print(type(vectors_features)) # list
    # print(type(y)) # list

    return vectors, vectors_features, y

def model_rfr(vectors: np.ndarray, vectors_features:list, labels:list, llm_do:int, features_exc:int, results_configuration:str) -> dict:
    """
    Runs the Random Forest Regressor

    Parameters
    -----------------------
    vectors: np.ndarray,
        the input X
    vectors_features: list,
        feature names of X
    labels: list,
        the input y
    llm_do: int,
        whether the dataset also contains data from the LLM
    features_exc: int,
        number of features excluded respect the original size
    results_configuration: str
        configuration directory for model dump

    Returns
    -----------------------
    Dictionary with model metrics
    """

    # sys.getsizeof in Bytes
    memory_size = sys.getsizeof(vectors) + sys.getsizeof(labels)

    print(">> Train / Test split")
    print()
    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2)

    # print(type(X_train)) # numpy.ndarray
    # print(type(y_train)) # list

    # get the number of features considered
    vectors_features_len = len(vectors_features)

    # get min and max values from y_test (the duration labels)
    min_test = min(y_test)
    max_test = max(y_test)
    print("Min value y_test:", min_test)
    print("Max value y_test:", max_test)
    print()

    print(">> Trainining (fit) - {}".format(model_suffix))
    print()

    rf = RandomForestRegressor(n_jobs=-1) # n_jobs=-1 -> use all processors
    # Fit the model to the data
    rf.fit(X_train, y_train)

    print(">> Predicting - {}".format(model_suffix))
    print()

    start_time_model = time.time()

    y_pred = rf.predict(X_test)

    # print(type(y_test)) # list
    # print(type(y_pred)) # numpy.ndarray

    print(">> Metrics")
    print()
    
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / np.abs(y_test)))
    round_mape = round(mape * 100, 2)
    #accuracy = round(100*(1 - mape), 2)
    # Normalized RMSE = RMSE / (max value – min value)
    rmse_n = rmse / (max_test - min_test)

    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse) # rmse = mse**.5
    print('Root Mean Squared Error (RMSE) normalized:', rmse_n)
    print('Mean Absolute Percentage Error (MAPE):', round_mape)
    # print('Accuracy:', accuracy)
    print()

    # HT
    print(">> Hyperparameters Tuning - {}".format(model_suffix))
    print()
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_features': ['sqrt','log2'],
        'max_depth' : [None, 10, 20],
        'min_samples_split': [2, 4, 8] # the minimum value is 2
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv_folds_num, n_jobs=-1)

    # Fit the HT model to the data
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_rf = grid_search.best_estimator_

    # Make new prediction with HT model
    y_pred = best_rf.predict(X_test)

    end_time_model = time.time()

    time_delta_model = int(end_time_model - start_time_model)

    mae_ht = metrics.mean_absolute_error(y_test, y_pred)
    mse_ht = metrics.mean_squared_error(y_test, y_pred)
    rmse_ht = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    mape_ht = np.mean(np.abs((y_test - y_pred) / np.abs(y_test)))
    round_mape_ht = round(mape_ht * 100, 2)
    # accuracy_ht = round(100*(1 - mape_ht), 2)
    rmse_ht_n = rmse_ht / (max_test - min_test)

    print('Mean Absolute Error (MAE) with HT:', mae_ht)
    print('Mean Squared Error (MSE) with HT:', mse_ht)
    print('Root Mean Squared Error (RMSE) with HT:', rmse_ht) # rmse = mse**.5
    print('Root Mean Squared Error (RMSE) with HT normalized:', rmse_ht_n)
    print('Mean Absolute Percentage Error (MAPE) with HT:', round_mape_ht)
    # print('Accuracy with HT:', accuracy_ht)

    string_hours = seconds_to_hours(time_delta_model)

    print()
    print("- Timing:", time_delta_model, "(",string_hours,")")
    print()

    # Save metrics in the dictionary
    dic_result = {'file_name':file_csv, 'prefix_length': prefix_len, 'prefix_encoding': prefix_enc, 'model': model_suffix, 'features_num': vectors_features_len, 'features_excluded': list_col_exclude_len, 'LLM_data': llm_do, 'CV': cv_folds_num, 'min_duration_dd':min_test, 'max_duration_dd': max_test, 'RMSE_ht': rmse_ht, 'RMSE_ht_norm': rmse_ht_n, 'MAE_ht': mae_ht, 'MSE_ht': mse_ht, 'MAPE_ht': round_mape_ht, 'RMSE': rmse, 'RMSE_norm':rmse_n, 'MAE': mae, 'MSE': mse, 'MAPE': round_mape, 'timing_sec':time_delta_model, 'timing_hr':string_hours, 'memory': memory_size}

    # Save the model and its data
    if model_dump == 1:
        model_save(best_rf, model_suffix, prefix_enc, prefix_len, X_train, y_train, X_test, y_test, vectors_features, results_configuration)

    # model_shap(best_rf, model_suffix, prefix_enc, prefix_len, X_train, y_train, X_test, y_test, vectors_features)

    return dic_result

def model_xgb(vectors: np.ndarray, vectors_features:list, labels:list, llm_do:int, features_exc:int, results_configuration:str)-> dict:
    """
    Runs the Gradient Boosting Regressor

    Parameters
    -----------------------
    vectors: np.ndarray,
        the input X
    vectors_features:list,
        feature names of X
    labels: list,
        the input y
    llm_do: int,
        whether the dataset also contains data from the LLM
    features_exc: int,
        number of features excluded respect the original size
    results_configuration: str
        configuration directory for model dump

    Returns
    -----------------------
    Dictionary with model metrics
    """

    # sys.getsizeof in Bytes
    memory_size = sys.getsizeof(vectors) + sys.getsizeof(labels)

    print(">> Train / Test split")
    print()
    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2)

    # get the number of features considered
    vectors_features_len = len(vectors_features)

    # get min and max values from y_test (the duration labels)
    min_test = min(y_test)
    max_test = max(y_test)

    print("Min value y_test:", min_test)
    print("Max value y_test:", max_test)
    print()

    print(">> Trainining (fit)")
    print()

    xgb_model = xgb.XGBRegressor()
    # Fit the model to the data
    xgb_model.fit(X_train, y_train)

    print(">> Predicting")
    print()

    start_time_model = time.time()

    y_pred = xgb_model.predict(X_test)

    # print(type(y_test)) # list
    # print(type(y_pred)) # numpy.ndarray

    print(">> Metrics")
    print()
    
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / np.abs(y_test)))
    round_mape = round(mape * 100, 2)
    # accuracy = round(100*(1 - mape), 2)

    rmse_n = rmse / (max_test - min_test)

    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse) # rmse = mse**.5
    print('Root Mean Squared Error (RMSE) normalized:', rmse_n)
    print('Mean Absolute Percentage Error (MAPE):', round_mape)
    # print('Accuracy:', accuracy)
    print()

    # HT
    print(">> Hyperparameters Tuning")
    print()
    param_grid = {
                "colsample_bytree": uniform(0.7, 0.3),
                "gamma": uniform(0, 0.5),
                "learning_rate": uniform(0.03, 0.3), # default 0.1 
                "max_depth": randint(2, 6), # default 3
                "n_estimators": randint(100, 150), # default 100
                "subsample": uniform(0.6, 0.4)
    }

    grid_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, random_state=42, n_iter=200, cv=cv_folds_num, verbose=1, n_jobs=1, return_train_score=True)

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_xgb = grid_search.best_estimator_

    # report_best_scores(grid_search.cv_results_, 1)

    # Make new prediction with HT model
    y_pred = best_xgb.predict(X_test)

    end_time_model = time.time()

    time_delta_model = int(end_time_model - start_time_model)

    mae_ht = metrics.mean_absolute_error(y_test, y_pred)
    mse_ht = metrics.mean_squared_error(y_test, y_pred)
    rmse_ht = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    mape_ht = np.mean(np.abs((y_test - y_pred) / np.abs(y_test)))
    round_mape_ht = round(mape_ht * 100, 2)
    # accuracy_ht = round(100*(1 - mape_ht), 2)
    # Normalized RMSE = RMSE / (max value – min value)
    rmse_ht_n = rmse_ht / (max_test - min_test)

    print('Mean Absolute Error (MAE) with HT:', mae_ht)
    print('Mean Squared Error (MSE) with HT:', mse_ht)
    print('Root Mean Squared Error (RMSE) with HT:', rmse_ht) # rmse = mse**.5
    print('Root Mean Squared Error (RMSE) with HT normalized:', rmse_ht_n)
    print('Mean Absolute Percentage Error (MAPE) with HT:', round_mape_ht)
    # print('Accuracy with HT:', accuracy_ht)

    string_hours = seconds_to_hours(time_delta_model)

    print()
    print("- Timing:", time_delta_model, "(",string_hours,")")
    print()

    # Save metrics in the dictionary
    dic_result = {'file_name':file_csv, 'prefix_length': prefix_len, 'prefix_encoding': prefix_enc, 'model': model_suffix, 'features_num': vectors_features_len, 'features_excluded': features_exc, 'LLM_data': llm_do, 'CV': cv_folds_num, 'min_duration_dd':min_test, 'max_duration_dd': max_test, 'RMSE_ht': rmse_ht, 'RMSE_ht_norm': rmse_ht_n, 'MAE_ht': mae_ht, 'MSE_ht': mse_ht, 'MAPE_ht': round_mape_ht, 'RMSE': rmse, 'RMSE_norm':rmse_n, 'MAE': mae, 'MSE': mse, 'MAPE': round_mape, 'timing_sec':time_delta_model, 'timing_hr':string_hours, 'memory': memory_size}

    # Save the model and its data
    if model_dump == 1:
        model_save(best_xgb, model_suffix, prefix_enc, prefix_len, X_train, y_train, X_test, y_test, vectors_features, results_configuration)

    # model_shap(best_xgb, model_suffix, prefix_enc, prefix_len, X_train, y_train, X_test, y_test, vectors_features)

    return dic_result

def model_shap(best_model, model_suffix, prefix_enc, prefix_len, X_train, y_train, X_test, y_test, vectors_features):
    """    
    Create the SHAP figure. Remember the SHAP model is built on the training data set
    """
    print(">> Creating SHAP plot")
    
    # SHAP computing
    shap_explainer = shap.TreeExplainer(best_model)
    shap_values = shap_explainer.shap_values(X_train)
    
    # name of the SHAP model 
    shap_name = "".join([model_suffix,"_",str(prefix_enc),"_",str(prefix_len)]) 
    
    # Create the SHAP values and save it to dataframe, csv and xlsx
    model_shap_df = pd.DataFrame(shap_values, columns = vectors_features)
    vals = np.abs(model_shap_df.values).mean(0)
    shap_importance_df = pd.DataFrame(list(zip(vectors_features, vals)),columns=['feature_name','feature_importance_value'])
    shap_importance_df.sort_values(by=['feature_importance_value'], ascending=False, inplace=True)
    # print(shap_importance.head()) # debug
    shap_file = "".join([shap_name,".csv"])
    path_shap = os_path_join(dir_models_shap, shap_file)
    print("Saving SHAP values to:", path_shap)
    shap_importance_df.to_csv(path_shap, sep=";")
    shap_file = "".join([shap_name,".xlsx"])
    path_shap = os_path_join(dir_models_shap, shap_file)
    print("Saving SHAP values to:", path_shap)
    shap_importance_df.to_excel(path_shap, sheet_name=shap_name)

    # Create the SHAP plot and save it to figure
    shap_file = "".join([shap_name,".png"])
    path_shap = os_path_join(dir_models_shap, shap_file)

    fig = plt.figure()
    plt.title(shap_name) 
    shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names = vectors_features, show=False) # show=False to avoid onscreen figure
    # plt.show()
    print("Saving SHAP to:", path_shap)
    plt.savefig(path_shap, bbox_inches='tight', dpi=600)
    plt.clf()
    print()

def model_save(model_object, model_suffix, prefix_enc, prefix_len, xtrain, ytrain, xtest, ytest, feature_names, results_configuration):
    """
    Given a model, its suffix and test data, it saves it in binary
    """
    # model
    joblib_file = "".join([model_suffix,"_",str(prefix_enc),"_",str(prefix_len),"_","model",".bin"])
    path_joblib = os_path_join(dir_models_dump, results_configuration, joblib_file)
    print("Saving model to:", path_joblib)
    joblib.dump(model_object, path_joblib)
    # X_train
    joblib_file = "".join([model_suffix,"_",str(prefix_enc),"_",str(prefix_len),"_","xtrain",".bin"])
    path_joblib = os_path_join(dir_models_dump, results_configuration, joblib_file)
    print("Saving X_train to:", path_joblib)
    joblib.dump(xtrain, path_joblib)
    # y_train
    joblib_file = "".join([model_suffix,"_",str(prefix_enc),"_",str(prefix_len),"_","ytrain",".bin"])
    path_joblib = os_path_join(dir_models_dump, results_configuration, joblib_file)
    print("Saving y_train to:", path_joblib)
    joblib.dump(ytrain, path_joblib)
    # X_test
    joblib_file = "".join([model_suffix,"_",str(prefix_enc),"_",str(prefix_len),"_","xtest",".bin"])
    path_joblib = os_path_join(dir_models_dump, results_configuration, joblib_file)
    print("Saving X_test to:", path_joblib)
    joblib.dump(xtest, path_joblib)
    # y_test
    joblib_file = "".join([model_suffix,"_",str(prefix_enc),"_",str(prefix_len),"_","ytest",".bin"])
    path_joblib = os_path_join(dir_models_dump, results_configuration, joblib_file)
    print("Saving y_test to:", path_joblib)
    joblib.dump(ytest, path_joblib)
    # feature names
    joblib_file = "".join([model_suffix,"_",str(prefix_enc),"_",str(prefix_len),"_","x_names",".bin"])
    path_joblib = os_path_join(dir_models_dump, results_configuration, joblib_file)
    print("Saving feature names to:", path_joblib)
    joblib.dump(feature_names, path_joblib)
    print()

### MAIN ###

print()
print("*** PROGRAM START ***")
print()

start_time = datetime.now().replace(microsecond=0)
# print(type(start_time)) # debug

# app-log init
app_log_init(app_log_dir, app_log_file, app_log_file_header)

print("Starting time:", start_time)
print()

# Creation of output directories
print(">> Creating output directories")
print()
check_and_create_directory(app_log_dir)
check_and_create_directory(dir_models_results)
check_and_create_directory(dir_models_shap)
check_and_create_directory(dir_models_plot)
check_and_create_directory(dir_models_dump)
if model_dump == 1:
    check_and_create_directory(results_configuration, dir_models_dump)
print()

print(">> Configuration")
print("Prefix list:",prefix_list)
print("Columns excluded ({}): ".format(list_col_exclude_len, list_col_exclude))
print("LLM do:",llm_do)
print()

print(">> Versioning")
print("Python version:",sys.version)
print("Scikit-learn verion:", sklearn.__version__)
print("Joblib verion:", joblib.__version__)
print("SHAP version:", shap.__version__)
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

    result_rows = [] # list with result dictionaries

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

        if int(prefix_len) not in prefix_list:
            print("Prefix value not in list, skipped")
            print()
            continue

        # Reading CSV
        print(">> Reading:", file_csv)
        df_log = read_log_encoded(dir_log_encoded, file_csv, list_col_exclude)
        print("DF columns:")
        print(df_log.columns)
        # print(df_in.head()) # debug
        print()

        # Add th amount increase from log_llm
        if llm_do == 1:
            print(">> Getting LLM data:", path_llm)
            print()
            df_llm = get_llm_data(path_llm, csv_separator)
            print(df_llm.columns)
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

        print(">> Extracting data")
        print()
        
        # Adds columns to be scaled to the list
        list_col_scale.append(log_amount_col_name)

        if llm_do == 1:
            list_col_scale.append("tender_increase_label")

        # Extracts the matrix of feature values, future names and labels
        vectors, vectors_features, labels = log_encoded_extract_data(df_log, log_case_id_col_name, log_label_col_name, list_col_scale)

        dic_result = {}

        if model_suffix == yaml_config['RFR_SUFFIX']:
            print(">> Performing RFR")
            print()
            dic_result = model_rfr(vectors, vectors_features, labels, llm_do, list_col_exclude_len, results_configuration)

        if model_suffix == yaml_config['XGR_SUFFIX']:
            print(">> Performing XGB")
            print()
            dic_result = model_xgb(vectors, vectors_features, labels, llm_do, list_col_exclude_len, results_configuration)

        # Save dictionary in the list
        result_rows.append(dic_result)

        print()
        print("-"*6)
        print()

    # All regressions are completed
    print()
    print("Regression '{}' concluded".format(model_suffix))
    print()
    print()

    # Creates a dataframe from the result list and order it by order_metric
    df_results = pd.DataFrame.from_records(result_rows)
    df_results = df_results.sort_values(by = order_metric, ascending=True)
    # print(df_results.head()) # debug

    # Save metric results
    path_results = os_path_join(dir_models_results, file_results_csv)
    print("Saving results:", file_results_csv)
    df_results.to_csv(path_results, sep = csv_separator, index=False)

    path_results = os_path_join(dir_models_results, file_results_xlsx)
    print("Saving results:", file_results_xlsx)
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



