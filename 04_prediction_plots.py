# 02_log_encoding.py
# Prefixes encoding

### IMPORT ###
import os
import pandas as pd
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

### LOCAL IMPORT ###
from config.config_reader import ConfigReadYaml

### GLOBALS ###
yaml_config = ConfigReadYaml()
dir_model_results = yaml_config['DIR_RESULTS'] 
dir_model_plots = yaml_config['DIR_PLOT'] 
csv_separator = str(yaml_config['CSV_DEFAULT_SEPARATOR'])
order_metric = str(yaml_config['COL_ORDER_METRIC'])

rfr_prefix = str(yaml_config['RFR_SUFFIX'])
xgr_prefix = str(yaml_config['XGR_SUFFIX'])

models_prefix_list = []
models_prefix_list.append(rfr_prefix)
models_prefix_list.append(xgr_prefix)

prefix_list = [7, 9, 11] # List of prefixes whose boxplot is desired

models_suffix_dic = {rfr_prefix: "Random Forest Regressor", xgr_prefix: "Gradient Boosting Regression"}

### MAIN ###

start_time = datetime.now().replace(microsecond=0)

print()
print("*** PROGRAM START ***")
print()

print("Starting time:", start_time)
print()

print(">> Configuration")
print("Models:", models_prefix_list)
print("Matplotlib version:", matplotlib.__version__)
print("Seaborn version:", sns.__version__)
print()

print(">> Getting results")
for model_suffix in models_prefix_list:
    print("Model:", model_suffix)
    file_model = "".join(["regression_", model_suffix ,".csv"])
    print("Model result file:", file_model)
    path_model = os.path.join(dir_model_results, file_model)
    print("Result file:", path_model)
    df_xgb = pd.read_csv(path_model, sep = csv_separator, low_memory=False)
    df_xgb.sort_values(by = [order_metric], ascending=False)
    print(df_xgb.head())
    # print(df_xgb.columns) # debug
    print()

    df_xgb_plot = df_xgb[df_xgb['prefix_length'].isin(prefix_list)] # Gets only rows with the prefix defined in the list

    # @TODO
    # avg of the order_metric by encoding type (Series) and prefix length 
    for prefix_len in prefix_list:
        print("Prefix:", prefix_len)
        df_xgb_plot_p = df_xgb_plot[df_xgb_plot['prefix_length']==prefix_len]
        medians_p = df_xgb_plot_p.groupby(['prefix_encoding'])[order_metric].mean()
        print(medians_p)
        print()
    # print(medians) # debug
    # vertical_offset = df_xgb_plot[order_metric].mean() * 0.08 # offset from median for display
    # print(vertical_offset) # debug

    model_name = models_suffix_dic[model_suffix] # Complete model name

    sns_title = "".join(["Model: ", model_name, " ", "(", model_suffix, ")"])
    sns.set(style="darkgrid")
    sns.set(rc={'figure.figsize':(15.7,8.27)}) # figure size (w x h) in inches

    # order = order of the x
    # showmeans = show the mean dot
    # palette = palette of the boxes
    # width =  width of the boxes
    # gap =  space between boxes of the same group
    chart = sns.boxplot(x='prefix_encoding', y='RMSE_ht', hue='prefix_length', data=df_xgb_plot, order=["B", "F", "I"], showmeans=True, meanprops={"marker":"o","markerfacecolor":"white" ,"markersize":"10" ,"markeredgecolor":"black"}, palette="Blues", width=0.8, gap=.2)

    """
    for xtick in chart.get_xticks():
        print(xtick)
        chart.text(xtick, medians[xtick] + vertical_offset, medians[xtick], horizontalalignment='center',size='x-small',color='b',weight='semibold')
        # x, y, string_text
    """
    
    # title, x and y
    chart.set_title(sns_title, fontdict={'size': 20, 'weight': 'bold'})
    chart.set_xlabel('Encoding type', fontdict={'size': 11, 'weight': 'bold'})
    chart.set_ylabel('RMSE', fontdict={'size': 11, 'weight': 'bold'})
    
    # File where to save the plot
    file_plot = "".join([file_model.split(".")[0], ".png"])
    path_plot = os.path.join(dir_model_plots,file_plot)
    print("Saving plot to:", path_plot)

    # Position and title of the legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Prefix length')

    # Save and show
    plt.savefig(path_plot, bbox_inches='tight', format='png', dpi=150)
    plt.tight_layout()
    plt.show()

end_time = datetime.now().replace(microsecond=0)

print("End process:", end_time)
print()
print("Time to finish:", end_time - start_time)
print()

print()
print("*** PROGRAM END ***")
print()
