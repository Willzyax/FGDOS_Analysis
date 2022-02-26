f(flag_refplot):
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



    # read data into dataframe to perform calculations
    # first i lines, and an extra one (5) can all be skipped by pandas (hence the i + 5)
    # different dataframes are stored in a dictionary with keys numbers according to read order
    # data= pd.read_csv(folder+"\\"+fileName, skiprows = startrow, delimiter = ",")
    # data.columns = column_names
    # data[column_x] = (data[column_x].values/time_divisor).astype(int)
    # this changes the time to cumulative time, this is not needed in later datafiles
    # fgdos_data.iat[0,0] = 0
    # time = 0
    # for i in range(0,len(fgdos_data)):
    #     fgdos_data['time'][i] = fgdos_data['time'][i] + time
    #     time = fgdos_data['time'][i]

    # always close file or use with open() as: ... (this closes file at end of with automatically)

    
    # read data into dataframe to perform calculations
    # different dataframes are stored in a dictionary with keys numbers according to read order
    # time redefined and severly negative values = 0
    # added current and power columns (R_shunt = 0.1 ohm), overwrite csv file
    # add data stats summary to csv file 
    data= pd.read_csv(newfilename, skiprows = 0, delimiter = ",")
    data.columns = [t,shunt_info_1,v_shunt_1,bus_info_1,v_bus_1,shunt_info_2,v_shunt_2,bus_info_2,v_bus_2]
    data[t] = (data[t].values/time_divisor).astype(int)
    temp = data._get_numeric_data() 
    temp[temp < -24000] = 0
    data.insert(5,i_shunt_1,data[v_shunt_1]/R_shunt)
    data.insert(6,p_1,(data[v_bus_1]*data[i_shunt_1])/10**6)
    data.insert(11,i_shunt_2,data[v_shunt_2]/R_shunt)
    data.insert(12,p_2,(data[v_bus_2]*data[i_shunt_2])/10**6)
    data.to_csv(newfilename)

    return [file_name],data


def sens_calc_rate(dataset,freq_selection,dose_rate):
    """calculate the sensitivity of the dataset, given the dose rate and an array of time selection parameters
    that select which periods in the dataset to use as start and endpoints

    Args:
        dataset (pandas dataframe): data to analyse constructed as per convention
        time_selection (int array): 2 points to indicate start period and 2 points to indicate end period, alternatively min and max can be chosen
        dose_rate (float): dose rate in Gy/s
    """    
    F_s = dataset[F_s_info]
    t_s = dataset[t_info]
    length = len(freq_selection)
    if (freq_selection[0] == 'max' and freq_selection[1] == 'min'):
        F_start = np.max(F_s) 
        F_end = np.min(F_s)
        t_start = t_s[F_s == F_start].iat[0]
        t_end = t_s[F_s == F_end].iat[0]
    elif (freq_selection[0] == 'max' and length == 3):
        F_start = np.max(F_s) 
        F_end = np.mean(F_s.loc[freq_selection[1]:freq_selection[2]])
        t_start = t_s[F_s == F_start].iat[0]
        t_end = t_s[freq_selection[1]]
    elif (freq_selection[2] == 'min' and length == 3):
        F_start = np.mean(F_s.loc[freq_selection[0]:freq_selection[1]]) 
        F_end = np.min(F_s)
        t_start = t_s[freq_selection[1]]
        t_end = t_s[F_s == F_end].iat[0]
    elif (length == 4):
        F_start = np.mean(F_s.loc[freq_selection[0]:freq_selection[1]])  
        F_end = np.mean(F_s.loc[freq_selection[2]:freq_selection[3]])
        t_start = t_s[freq_selection[1]]
        t_end = t_s[freq_selection[2]]
    else:
        print('invalid time selection array')
        return -1
    # print('%f,%f,%f,%f'%(F_start,F_end,t_start,t_end))
    dt = (t_end-t_start)/1000 # s
    dF = F_start - F_end
    dose = dose_rate*dt
    return dF/dose

def sens_calc_dose(dataset,freq_selection,dose):
    """calculate the sensitivity of the dataset, given the dose  and an array of time selection parameters
    that select which periods in the dataset to use as start and endpoints

    Args:
        dataset (pandas dataframe): data to analyse constructed as per convention
        time_selection (int array): 2 points to indicate start period and 2 points to indicate end period, alternatively min and max can be chosen
        dose (float): dose in Gy
    """    
    F_s = dataset[F_s_info]
    length = len(freq_selection)
    if (freq_selection[0] == 'max' and freq_selection[1] == 'min'):
        F_start = np.max(F_s) 
        F_end = np.min(F_s)
    elif (freq_selection[0] == 'max' and length == 3):
        F_start = np.max(F_s) 
        F_end = np.mean(F_s.loc[freq_selection[1]:freq_selection[2]])
    elif (freq_selection[2] == 'min' and length == 3):
        F_start = np.mean(F_s.loc[freq_selection[0]:freq_selection[1]]) 
        F_end = np.min(F_s)
    elif (length == 4):
        F_start = np.mean(F_s.loc[freq_selection[0]:freq_selection[1]])  
        F_end = np.mean(F_s.loc[freq_selection[2]:freq_selection[3]])
    else:
        print('invalid time selection array')
        return -1
    # print('%f,%f,%f'%(F_start,F_end,dose))
    dF = F_start - F_end
    return dF/dose

    #  A4 HIGH
    #  determine sens via linreg over linear ranges
    x_1a, x_1b, x_1c = dataset_1[t_info].iloc[i_start_1_1_linear:i_end_1_1_linear], dataset_1[t_info].iloc[i_start_1_2_linear:i_end_1_2], (
                        dataset_1[t_info].iloc[i_start_1_3:i_end_1_2_linear] )
    x_2a, x_2b, x_2c = dataset_2[t_info].iloc[i_start_2_1_linear:i_end_2_1_linear], dataset_2[t_info].iloc[i_start_2_2_linear:i_end_2_2], (
                        dataset_2[t_info].iloc[i_start_2_3:i_end_2_2_linear] )
    y_1a, y_1b, y_1c = dataset_1[F_s_info].iloc[i_start_1_1_linear:i_end_1_1_linear], dataset_1[F_s_info].iloc[i_start_1_2_linear:i_end_1_2], (
                        dataset_1[F_s_info].iloc[i_start_1_3:i_end_1_2_linear] )
    y_2a, y_2b, y_2c = dataset_2[F_s_info].iloc[i_start_2_1_linear:i_end_2_1_linear], dataset_2[F_s_info].iloc[i_start_2_2_linear:i_end_2_2], (
                        dataset_2[F_s_info].iloc[i_start_2_3:i_end_2_2_linear] )                        
    x_1a_sm, x_1b_sm, x_1c_sm, x_2a_sm, x_2b_sm, x_2c_sm = sm.add_constant(x_1a), sm.add_constant(x_1b), sm.add_constant(x_1c), sm.add_constant(x_2a), (
                        sm.add_constant(x_2b)), sm.add_constant(x_2c)
    model_1a, model_1b, model_1c, model_2a, model_2b, model_2c = sm.OLS(y_1a,x_1a_sm).fit(), sm.OLS(y_1b,x_1b_sm).fit(), sm.OLS(y_1c,x_1c_sm).fit(), (
                        sm.OLS(y_2a,x_2a_sm).fit()), sm.OLS(y_2b,x_2b_sm).fit(), sm.OLS(y_2c,x_2c_sm).fit() 
    intercept_1a, coeff_1a, intercept_1b, coeff_1b, intercept_1c, coeff_1c = pd.concat([model_1a.params,model_1b.params,model_1c.params])
    intercept_2a, coeff_2a, intercept_2b, coeff_2b, intercept_2c, coeff_2c = pd.concat([model_2a.params, model_2b.params, model_2c.params])
    p_intercept_1a, p_coeff_1a, p_intercept_1b, p_coeff_1b, p_intercept_1c, p_coeff_1c = pd.concat([model_1a.pvalues,model_1b.pvalues,model_1c.pvalues]) 
    p_intercept_2a, p_coeff_2a, p_intercept_2b, p_coeff_2b, p_intercept_2c, p_coeff_2c = pd.concat([model_2a.pvalues,model_2b.pvalues,model_2c.pvalues]) 

        print("sensitivity over linear range (linreg) (S1a | S1b | S1c | S2a | S2b | S2c) (Hz/Gy): %f | %f | %f | %f | %f | %f"%(
                        sens_A4_HIGH_1a,sens_A4_HIGH_1b,sens_A4_HIGH_1c ,sens_A4_HIGH_2a,sens_A4_HIGH_2b,sens_A4_HIGH_2c))
    print("p-values of linreg fits for intercept and coeff (S1a | S1b | S1c | S2a | S2b | S2c) (Hz/Gy): %f , %f | %f , %f | %f , %f | %f , %f | %f , %f | %f , %f"%(
                p_intercept_1a, p_coeff_1a, p_intercept_1b, p_coeff_1b, p_intercept_1c, p_coeff_1c, p_intercept_2a, p_coeff_2a, p_intercept_2b, p_coeff_2b,
                p_intercept_2c, p_coeff_2c))
    sens_A4_HIGH_1a, sens_A4_HIGH_1b, sens_A4_HIGH_1c, sens_A4_HIGH_2a, sens_A4_HIGH_2b, sens_A4_HIGH_2c = -coeff_1a/dose_rate*1000 , -coeff_1b/dose_rate*1000, (
                        -coeff_1c/dose_rate*1000) , -coeff_2a/dose_rate*1000, -coeff_2b/dose_rate*1000, -coeff_2c/dose_rate*1000 # Hz/Gy
    print("sensitivity over linear range (linreg) (S1a | S1b | S1c | S2a | S2b | S2c) (Hz/Gy): %f | %f | %f | %f | %f | %f"%(
                        sens_A4_HIGH_1a,sens_A4_HIGH_1b,sens_A4_HIGH_1c ,sens_A4_HIGH_2a,sens_A4_HIGH_2b,sens_A4_HIGH_2c))
    print("p-values of linreg fits for intercept and coeff (S1a | S1b | S1c | S2a | S2b | S2c) (Hz/Gy): %f , %f | %f , %f | %f , %f | %f , %f | %f , %f | %f , %f"%(
                p_intercept_1a, p_coeff_1a, p_intercept_1b, p_coeff_1b, p_intercept_1c, p_coeff_1c, p_intercept_2a, p_coeff_2a, p_intercept_2b, p_coeff_2b,
                p_intercept_2c, p_coeff_2c))

    #  A3 HIGH sens over linear range via linreg
    x_1, x_2 = sm.add_constant(dataset_1.loc[(dataset_1[F_s_info] < F_lin_max) & (dataset_1[F_s_info] > F_lin_min) & (dataset_1.index > i_r_end_1)][t_info]), sm.add_constant(
                dataset_2.loc[(dataset_2[F_s_info] < F_lin_max) & (dataset_2[F_s_info] > F_lin_min) & (dataset_2[R_r_info] ==1)][t_info])
    y_1, y_2 = dataset_1.loc[(dataset_1[F_s_info] < F_lin_max) & (dataset_1[F_s_info] > F_lin_min) & (dataset_1.index > i_r_end_1)][F_s_info], (
                dataset_2.loc[(dataset_2[F_s_info] < F_lin_max) & (dataset_2[F_s_info] > F_lin_min) & (dataset_2[R_r_info] ==1)][F_s_info])

    model_1, model_2 = sm.OLS(y_1,x_1).fit(), sm.OLS(y_2,x_2).fit()
    intercept_1, coeff_1 = model_1.params
    intercept_2, coeff_2 = model_2.params
    p_intercept_1, p_coeff_1 = model_1.pvalues 
    p_intercept_2, p_coeff_2 = model_2.pvalues
    sens_A3_HIGH_1, sens_A3_HIGH_2 = -coeff_1/dose_rate*1000 , -coeff_2/dose_rate*1000 # Hz/Gy
    print("sensitivity over linear range (linear regression) (S1 | S2) (Hz/Gy): %f | %f"%(sens_A3_HIGH_1,sens_A3_HIGH_2))
    print("p-values of linreg fits for intercept and coeff (S1 | S2): %f , %f | %f , %f"%( p_intercept_1, p_coeff_1,p_intercept_2, p_coeff_2))