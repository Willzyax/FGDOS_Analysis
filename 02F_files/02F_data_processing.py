# TO ADD
# - add different temperature compensations
# - make sure reading either stops at a recharge or takes reacharging into account in another way
# - the test facility will provide data concerning radiation, so here code can be added to integrate the received dose into the dataframes
#   post - processing
#       - calculate the dose by performing a linear fit for both frequency and dose and compare total dose to total frequency change
#       - using this sensitivity other files can be analysed and changes in sensitvity can be analysed

from data_processing_functions_02F import *

from time import sleep
from numpy.core import numeric
import pandas as pd
import numpy as np
import math
import os
import seaborn as sb
import matplotlib.pyplot as plt
from statsmodels import stats
import statsmodels.api as sm
from scipy import stats as ss

# variables for dataset column names
t = "Time (s)"
s = "Sensor"
T = "Temperature (°C)"
Fs = "Sensor Frequency (Hz)"
Fr = "Reference Frequency (Hz)"
Rr = "Recharge Register"
column_names = [t,s , T ,Fs, Fr, Rr]
# set flags for what you want to do
flag_tempplot = True
flag_refplot = True
flag_sensplot = True
flag_linreg = False
flag_dist_conf = True
flag_ttest = False

def main():
    # get csv files in subfolder, set the start of filename parameter to select specific files
    start_of_filename = "FGDOS_02F_Nov_02"
    folder = "CSV_files"
    files = [f for f in os.listdir(os.getcwd()+"\\"+folder) if (f.endswith(".csv") and f.startswith(start_of_filename))]
    data_dict = {}
    file_names = [[]]*len(files)
    sensor_settings = [[]]*len(files)
    j = 0
    for file in files:
        file_names[j], data_dict["FGDOS_02F_file_%s" %j],sensor_settings[j]  = read_file(folder,file,column_names,t)
        j += 1

    if(flag_refplot):
        i = 0
        plt.ion()
        for dataset in data_dict.values() :
            plot_dataset(dataset,file_names[i][0],Fr,t)
            i += 1
    
    if(flag_tempplot):
        i = 0
        plt.ion()
        for dataset in data_dict.values() :
            plot_dataset(dataset,file_names[i][0],T,t)
            i += 1

    if(flag_sensplot):
        i = 0
        plt.ion()
        for dataset in data_dict.values() :
            plot_dataset(dataset,file_names[i][0],Fs,t)
            i += 1       

    if (flag_linreg):
        # SEABORN PLOTTING & STATSTOOLS LINEAR REGRESSION: plot linear regression and determine endpoints
        i = 0
        plt.ion()
        for dataset in data_dict.values() :
            plot_lin_reg(dataset,file_names[i][0],Fs,t)
            i += 1

    if (flag_dist_conf):
        # STATSTOOLS & SCIPY ANALYSIS: determine basic stats, plot hist and distributions, determine 95% confidence interval (based upon student t for now)
        plt.ion()
        i = 0
        conf_intervals = [[]]*len(data_dict)
        for dataset in data_dict.values() :
            # NOISE
            # first check to see if indeed normal distribution (Anderson-Darling statistic should be above crit value depending on confidence level)
            # then determine stat parameters and draw histogram with normal distr
            ad_stat, crit_val, alpha = ss.anderson(dataset[Fs],dist='norm')
            if (ad_stat>crit_val[2]): # 2nd crit for 95% confidence
                print("ANDERSON DARLING TEST OK (level: ",alpha[2],"%)")
                conf_intervals[i] = plot_stats_normal_student(dataset,file_names[i][0],Fs)    
            else:
                print("ANDERSON DARLING TEST FAILED")
            i += 1

    if (flag_ttest):
        # SCIPY ANALYSIS & T TESTS
        datasets = data_dict.values()
        t_tests_2sample(datasets,Fs)
    
    input("<Hit enter to close>\n")

if __name__ == "__main__":
    main()