# PREDICTIVE-PROCESS-MINING

## app_log
Log about scripts duration

## config  
Various application parameters

## log_disco  
Event-log filtered and exported from DISCO

## log_disco_duration  
DISCO stats file with cases duration (label)

## log_encoded  
The encoded prefix files

## log_input  
The event-log to use for prediction (to be used to extract prefixes and encode them)

## log_llm
The data obtained from LLM to improved the log

## log_prefix
The event-log prefixes

## log_stats
The log_input stats

## models_dump
The binary dump of the models applied to log prexifes (encoded)

## models_shap
SHAP values from models

## results_models
Results from models (03_log_prediction_RFR_XGR.py and 03_log_prediction_LSTM.py)

## results_models_plot
Plots of results from models

## 00_log_disco_duration.py
For each log filtered in DISCO, it adds the duration from the DISCO statistics and the partial timing (using ./log_disco and ./log_disco_duration)

## 01_log_prefixes.py
Extract the prefixes

## 02_log_encoding.py
Encode the prefixes in 3 modes; "B" for binary (output: log_2016-2022_clean_5_eventi_FORNITURE_P_1_*B*), F for frequency (output: log_2016-2022_clean_5_eventi_FORNITURE_P_1_*F*), I for simple-index (output: log_2016-2022_clean_5_eventi_FORNITURE_P_1_*I*)

## 03_log_prediction_RFR_XGR.py
Performs prediction with 2 machine learning algorithms: Random Forest Regressor (RFR) and Gradient Boosting Regression (XGR)

## 03_log_prediction_LSTM.py
Performs prediction with LSTM

## requirements.txt
Requirements file