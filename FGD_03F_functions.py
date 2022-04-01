from cmath import nan
import pandas as pd
import numpy as np
import math
import os
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats as ss
from scipy import signal as ssig
import fitter as ft

# radiation test variables
A1, A2, A3, A4 = [70,1.71*10**6],[70,7.07*10**6],[70,1.4*10**7],[70,3.69*10**7] # MeV, particles/cm²/s
LET_70_Si = 0.007641 # MeV cm²/mg, dE/dx electronic, SRIM
rad_conv = 1.6*10**-5 # conversion factor to rad
gy_conv = 1.6*10**-7 # conversion factor to Gy

# linear regression function definition
def f_linreg(a,b,x):
    return np.multiply(a,x)+b

# moving average definition
def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, 'same') 

# variables for dataset column names
sensor_info = "Sensor"
t_info = "Time (ms)"
T_info = "Temperature (\u00b0 C)"
F_s_info = "Sensor Frequency (Hz)"
F_r_info = "Reference Frequency (Hz)"
R_r_info = "Recharge Register"
wf_info = "Window Factor"
sens = "Sensitivity"
P_info = "Passive"
S_info = "Standby"
column_names = [t_info,sensor_info,T_info,F_s_info,F_r_info,R_r_info,wf_info,sens,P_info,S_info]
columns_string = t_info+','+sensor_info+','+T_info+','+F_s_info+','+F_r_info+','+R_r_info+','+wf_info+','+sens+','+P_info+','+S_info

# set matplotlib label size
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'font.weight': 555})
plt.rcParams.update({'font.family': 'sans-serif'})

def modify_file(folder,file,columns_string):
    """    opens a file in a folder, reads it according to predetermined structure, return data (filename, data, statistics)
    Args:
        folder (string): folder where files are stored
        file (string): filename
        column_names (string): column names
    Returns:
        [type]: [description]
    """
    print("---------------------READ & MODIFY FILE---------------------------")
    file_name = str(file)
    print(file_name)
    have_settings_s1 = False
    have_settings_s2 = False
    first_run = True
    F_s1, F_s2 = 0, 0
    # read first line, check to see if it is the sensor number, if not keep on reading lines until sensor number encountered
    # next read all the settings, note that this order is defined by Arduino programming
    # double values occuring after each other are deleted (are caused by an Arduino programming error)
    # are not tracked and deleted: resets after switching on the sensor, auto recharge on or off settings
    with  open(folder+"\\"+file_name, "r",newline='') as original:
        with open(folder+'\\corrected\\'+file_name[0:-4]+'_s1.csv',"w",newline='') as new_s1:
            with open(folder+'\\corrected\\'+file_name[0:-4]+'_s2.csv',"w",newline='') as new_s2:
                new_s1.write(columns_string+'\n')
                new_s2.write(columns_string+'\n')
                for line in original:
                    data_line = line.split(",")
                    # print(data_line[-1].strip(' ')[0:11])
                    if (len(data_line)<2):
                        pass
                    elif (len(data_line)>=6 and have_settings_s1 and have_settings_s2):
                        if (data_line[1].strip(' ')=='1'):
                            if (not first_run):
                                t_1 = (float(data_line[0].strip(' ')) - t_start)/1000 # to ms
                            else:
                                t_start = float(data_line[0].strip(' '))
                                t_1 = 0
                                first_run = False
                            F_s1_temp = int(data_line[3].strip(' '))
                            if (F_s1_temp != F_s1):
                                T_1 = int(data_line[2].strip(' '))
                                F_s1 = F_s1_temp
                                F_r1 = int(data_line[4].strip(' '))
                                R_r1 = int(data_line[5].strip(' '),2)
                                new_s1.write("%f,1,%d,%d,%d,%d,%f,%s,%d,%d \n"%(t_1,T_1,F_s1,F_r1,R_r1,
                                                window_factor_1,sensitivity_1,passive_1,standby_1))
                            else:
                                pass
                        elif (data_line[1].strip(' ')=='2'):
                            F_s2_temp = int(data_line[3].strip(' '))
                            if (F_s2_temp != F_s2):
                                t_2 = (float(data_line[0].strip(' ')) - t_start)/1000 # to ms
                                T_2 = int(data_line[2].strip(' '))
                                F_s2 = F_s2_temp
                                F_r2 = int(data_line[4].strip(' '))
                                R_r2 = int(data_line[5].strip(' '),2)
                                new_s2.write("%f,2,%d,%d,%d,%d,%f,%s,%d,%d \n"%(t_2,T_2,F_s2,F_r2,R_r2,
                                                window_factor_2,sensitivity_2,passive_2,standby_2))
                            else:
                                pass
                    elif (data_line[-1].strip(' ')[0:6] == "SENSOR"):
                        sensor_number = int(data_line[-1].split(" ")[-1])
                        if (sensor_number == 2):
                            try: 
                                window_factor_2 = float(next(original).split(" ")[-1][:-1])
                            except:
                                window_factor_2 = float(next(original).split(" ")[-1][:-1])
                            sensitivity_2 = next(original).split(" ")[-1][:1]
                            target_2 = float(next(original).split(",")[-1][1:-1])
                            threshold_2 = float(next(original).split(",")[-1][1:-1])
                            passive_2 = 0
                            standby_2 = 0
                            have_settings_s2 = True
                        if (sensor_number == 1):
                            try:
                                window_factor_1 = float(next(original).split(" ")[-1][:-1])
                            except:
                                window_factor_1 = float(next(original).split(" ")[-1][:-1])
                            sensitivity_1 = next(original).split(" ")[-1][:1]
                            target_1 = float(next(original).split(",")[-1][1:-1])
                            threshold_1 = float(next(original).split(",")[-1][1:-1])
                            passive_1 = 0
                            standby_1 = 0
                            have_settings_s1 = True
                        next(original)
                    elif (data_line[-1].strip(' ')[0:7] == "PASSIVE"):
                        passive_2 = 1
                        input("<PASSIVE. Hit enter to continue>\n")
                    elif (data_line[-1].strip(' ')[0:7] == "STANDBY"):
                        standby_2 = 1
                        input("<STANDBY. Hit enter to continue>\n")
                    elif (data_line[-1].strip(' ')[0:6] == "ACTIVE"):
                        passive_2 = 0
                        input("<ACTIVE. Hit enter to continue>\n")
                    elif (data_line[-1].strip(' ')[0:2] == "ON"):
                        standby_2 = 0
                        input("<ON. Hit enter to continue>\n")
                    elif (data_line[-1].strip(' ')[0:11] == 'measurement'):
                        # recharge ongoing (measurement loop running)
                        pass
                    elif (data_line[-1].strip(' ')[0:4] == 'Read'):
                        input("<READ ALL REGISTERS. Hit enter to continue>\n")
                    elif ("-----" in line):
                        input("<START FOUND. Hit enter to continue>\n")
                    elif(not have_settings_s1 and not have_settings_s1):
                        input('still looking for start...')
                        print(data_line)
                        # print(data_line[-1].strip(' ')[0:5])
                    else:
                        print('unknown condition', end=" ")
                        input(data_line)


    return file_name

def read_file(folder,file):
    """ 
    opens a corrected file in a folder, reads it according to predetermined structure, return all necessary data (filename, datastructure)
    """
    print("---------------------READ FILE---------------------------")
    file_name = str(file)
    file_folder = folder+'\\'+file_name
    print(file_name)
    # read data into dataframe to perform calculations
    data = pd.read_csv(file_folder, skiprows = 0, delimiter = ",",encoding='ISO-8859-1')

    return data

def max_change(dataset,start,end,variable = F_s_info):
    F_temp = dataset[variable][start]
    dF = 0  # max frequency change
    for F in dataset[variable][start:end]:
        if (abs(F-F_temp)> dF): 
            dF = abs(F-F_temp) 
        F_temp = F
    return dF

def limits(lists):
    lmin,lmax = math.floor(min(lists[0])), math.ceil(max(lists[0]))
    for list in lists:
        temp_min,temp_max = math.floor(min(list)), math.ceil(max(list))
        if (temp_min<lmin): lmin = temp_min
        if (temp_max>lmax): lmax = temp_max
    return lmin,lmax

def drop_shootouts(dataset,start,end,max,variable = F_s_info):
    dataset_new = dataset[:][start:end]
    indeces = dataset_new.index.tolist()
    x = dataset_new[variable][indeces[0]]
    i = 0
    while i<(len(indeces)-1):
        temp = dataset_new[variable][indeces[i]]
        j = 1
        if(abs(temp-x)>max):
            while(abs(temp-x)>(j*max)):
                dataset_new.drop(indeces[i],axis=0,inplace=True)
                i+=1
                j+=2
                temp = dataset_new[variable][indeces[i]]
            x = temp
        else:
            x = temp
            i+=1
    return dataset_new

def plot_datasets(x_1,x_2,y_1,y_2,filename,steps,xlim=[-1,-1],ylim=[-1,-1],savefolder=os.getcwd()+"\\Figures"):
    """function to plot two x,y datasets, save the figures and return the fig,ax couple

    Args:
        x_1 (pandas list): [description]
        x_2 (pandas list): [description]
        y_1 (pandas list): [description]
        y_2 (pandas list): [description]
        filename (string): [description]
        steps (int): [description]
        xlim (list, optional): x range limits. Defaults to [-1,-1].
        ylim (list, optional): y range limits. Defaults to [-1,-1].
        savefolder (string, optional): folder path to save files. Defaults to os.getcwd()+"\Figures".

    Returns:
        [figure,axis]: [matplotlib figure and axes]
    """
    fig, ax = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_1, y=y_1,color='red',label = 'Sensor 1')
    sb.scatterplot(x=x_2, y=y_2,color='blue', label = 'Sensor 2')
    xmin, xmax= math.floor(min([min(x_1),min(x_2)])),math.ceil(max([max(x_1),max(x_2)]))
    step = round((xmax-xmin)/steps)
    ax.set_xticks(range(xmin,xmax,step))
    ax.set_xticklabels(range(xmin,xmax,step),rotation=40)
    if (xlim[0] != -1):
        ax.set_xlim(xlim)
    if (ylim[0] != -1):
        ax.set_ylim(ylim)

    ax.legend(loc='lower right')
    fig.suptitle(filename)
    plt.grid()
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()
    fig.savefig(savefolder+'\\'+filename+'.png') # this overwrites existing files!  

    return fig,ax 

def plot_xy(x,y,filename,names,steps,xlim=[-1,-1],ylim=[-1,-1],savefolder=os.getcwd()+"\\Figures",colors=['red','blue','green','orange','olive','cyan'],
            markers = ['o','o','o','o','o','o']):
    """function to plot any number of x,y  datasets, save the figure and return the matplotlib figure and axis

    Args:
        x (list of arrays): [description]
        y (list of arrays): [description]
        filename ([type]): [description]
        filename (string): [description]
        steps (int): [description]
        xlim (list, optional): x range limits. Defaults to [-1,-1].
        ylim (list, optional): y range limits. Defaults to [-1,-1].
        savefolder (string, optional): folder path to save files. Defaults to os.getcwd()+"\Figures".

    Returns:
        [figure,axis]: [matplotlib figure and axes]
    """
    fig, ax = plt.subplots(figsize=(15,7))
    xmin, xmax = -1,-1
    for i in range(len(x)):
        sb.scatterplot(x=x[i], y=y[i],color=colors[i],label = names[i],marker=markers[i])
        x_min_temp, x_max_temp = math.floor(min(x[i])),math.ceil(max(x[i]))
        if (x_min_temp < xmin or xmin == -1): xmin = x_min_temp
        if (x_max_temp > xmax or xmax == -1): xmax = x_max_temp

    step = round((xmax-xmin)/steps)
    ax.set_xticks(range(xmin,xmax,step))
    ax.set_xticklabels(range(xmin,xmax,step),rotation=40)
    if (xlim[0] != -1):
        ax.set_xlim(xlim)
    if (ylim[0] != -1):
        ax.set_ylim(ylim)

    ax.legend(loc='lower right')
    # ax.set_title('test')
    # fig.suptitle(filename)
    plt.grid()
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    # ax.set_visible(False)
    fig.show()
    fig.savefig(savefolder+'\\'+filename+'.png') # this overwrites existing files!  

    return fig,ax 

def plot_2subs_datasets(x_1,x_2,y_1,y_2,filename,names,steps,xlim=[-1,-1],ylim_1=[-1,-1],ylim_2=[-1,-1],savefolder=os.getcwd()+"\\Figures",colors=['red','blue'],
            markers = ['o','o']):
    """function to plot two x,y datasets in two subplots, save the figures and return the fig,ax couple

    Args:
        x_1 (pandas list): [description]
        x_2 (pandas list): [description]
        y_1 (pandas list): [description]
        y_2 (pandas list): [description]
        filename (string): [description]
        steps (int): [description]
        xlim (list, optional): x range limits. Defaults to [-1,-1].
        ylim (list, optional): y range limits. Defaults to [-1,-1].
        savefolder (string, optional): folder path to save files. Defaults to os.getcwd()+"\Figures".
        colors (list of strings, optional): colors for plots

    Returns:
        [figure,axis]: [matplotlib figure and axes]
    """
    fig, axs = plt.subplots(nrows=2,figsize=(17,8))
    sb.scatterplot(x=x_1, y=y_1,label = names[0], ax= axs[0], color=colors[0], marker = markers[0])
    sb.scatterplot(x=x_2, y=y_2, label = names[1], ax= axs[1], color=colors[1], marker = markers[1])
    if (xlim[0]==-1):
        xmin_1, xmax_1= limits([x_1,x_2])
        xmin_2, xmax_2 = limits([x_1,x_2])
    if (ylim_1[0]!=-1):
        axs[0].set_ylim(ylim_1)
    if (ylim_2[0]!=-1):
        axs[1].set_ylim(ylim_2)
    step_1, step_2 = round((xmax_1-xmin_1)/steps), round((xmax_2-xmin_2)/steps)
    axs[0].set_xticks(range(xmin_1,xmax_1,step_1)), axs[1].set_xticks(range(xmin_2,xmax_2,step_2)) 
    axs[0].set_xticklabels(range(xmin_1,xmax_1,step_1),rotation=40), axs[1].set_xticklabels(range(xmin_2,xmax_2,step_2),rotation=40)
    axs[0].ticklabel_format(axis='y' ,useOffset=False)

    axs[0].legend(loc='lower right'), axs[1].legend(loc='lower right')
    # fig.suptitle(filename)
    axs[0].grid(), axs[1].grid()
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()
    fig.savefig(savefolder+'\\'+filename+'.png') # this overwrites existing files!  

    return fig,axs     

def sens_linreg(x: str,y: str,index_ranges: list,datasets: object):
    """determine the sensitivity of the given pandas datasets within the ranges given by the indexes using the columns x and y via linear regression methods.

    Args:
        x (str): x axis data column name
        y (str): y axis data column name
        index_ranges (list): limits to determine over which regions in the dataset linear regression should be used
        datasets (object): pandas dataframe

    Returns:
        _type_: the intercepts, coefficients, p-values and R-squared values of the linear regression lines
    """    
    x_matrix = np.zeros((len(datasets),len(index_ranges)),dtype=object)
    y_matrix = np.zeros((len(datasets),len(index_ranges)),dtype=object)
    models = np.zeros((len(datasets),len(index_ranges)),dtype=object)
    intercepts = np.zeros((len(datasets),len(index_ranges)))
    coeffs = np.zeros((len(datasets),len(index_ranges)))
    rsquared = np.zeros((len(datasets),len(index_ranges)))
    pvalues = np.zeros((len(datasets),len(index_ranges)*2))
    i,j = 0,0
    for dataset in datasets:
        for range in index_ranges:
            x_matrix[j][i] = sm.add_constant(dataset[x].loc[range[0+2*j]:range[1+2*j]])
            y_matrix[j][i] = dataset[y].loc[range[0+2*j]:range[1+2*j]]
            models[j][i] = sm.OLS(y_matrix[j][i],x_matrix[j][i]).fit()
            intercepts[j][i], coeffs[j][i] = models[j][i].params
            rsquared[j][i] = models[j][i].rsquared
            pvalues[j][i*2], pvalues[j][i*2+1] = models[j][i].pvalues
            i+=1
        j+=1
        i = 0

    return intercepts,coeffs,pvalues,rsquared

def hist_dist_plot(data,columnname,filename,name,colors=['indianred','royalblue'],savefolder=os.getcwd()+"\\Figures",bins = 20,dist = 'none'):
    """plot a histogram and fit a distribution over it

    Args:
        data (pandas series): data to use
        columnname (string): title of the data series used for the histogram
        filename (string): filename to give names to the plot and save files
        name (string): label to give the data in the histogram
        colors (list, optional): Colours of the data in the plots. Defaults to ['indianred','royalblue'].
        savefolder (string, optional): Folder where to save files. Defaults to os.getcwd()+"\Figures".
        bins (int, optional): Number of bins to use in the histogram. Defaults to 20.
        dist (str, optional): Type of distribution to fit over histogram, options are none, normal and KDE. Defaults to 'none'.

    Returns:
        _type_: _description_
    """    
    fig, ax = plt.subplots(figsize=(15,7))
    plt.hist(data,bins=bins,density=True,color=colors[0],label=name, alpha  = 0.6)
    xmin, xmax = limits([data])
    x = np.linspace(xmin, xmax, 500)
    
    if (str(dist)=='none'):
        dist = nan
        param_1, param_2 = nan, nan
    elif (str(dist)=='normal'):
        mu, std = ss.norm.fit(data)
        p = ss.norm.pdf(x, mu, std)
        plt.plot(x, p, linewidth=2, color = colors[1],label = 'Normal Distribution')
        param_1,param_2 = mu, std
        dist = p
    elif (str(dist)=='kde'):
        kde = ss.gaussian_kde(data)
        plt.plot(x,kde.evaluate(x), color = colors[1], label = 'Gaussian KDE')
        param_1,param_2 = kde.factor, kde.covariance
        dist = kde
    ax.legend(loc='upper left')
    ax.grid()
    ax.set_xlabel(columnname)
    # fig.suptitle(filename)
    fig.tight_layout()
    fig.show()
    fig.savefig(savefolder+'\\'+filename+'.png')

    return [fig,ax,dist,param_1,param_2]

def kde_limits(data,steps,kde,confidence = 0.9545):
    """given data and a KDE fit, determine endpoints of the interval determined by confidence

    Args:
        data (pandas series): data to use
        steps (int): number of data points to use to go over the KDE
        kde (Scipy model): Kernel Density Estimate Scipy model
        confidence (float, optional): Percentage of surface area under KDE to include. Defaults to 0.9545.

    Returns:
        list: endpoints of the calculated area
    """    
    kde_min, kde_max = (1-confidence)/2, confidence+(1-confidence)/2
    ecdf = sm.distributions.ECDF(data)
    xmin, xmax = limits([data])
    extension = (xmax-xmin)/4
    x = np.linspace(xmin-extension, xmax+extension, steps)
    low_lim,high_lim = -1000,-1000
    flag = True
    for x in x:
        kde_int = kde.integrate_box_1d(xmin,x)
        if (kde_int>=kde_min and kde_int<=kde_max and flag):
            print("Gaussian KDE lower noise range limit (confidence = %f | CDF = %f) (CDF | Hz) (S1): %f | %f "%(confidence,kde_min,kde_int,x))
            print("ECDF double check: %f"%(ecdf(x)))
            low_lim = x
            flag = False
        if(kde_int>=kde_max and not flag):
            print("Gaussian KDE higher noise range limit (confidence = %f | CDF = %f) (CDF | Hz) (S1): %f | %f "%(confidence,kde_max,kde_int,x))
            print("ECDF double check: %f"%(ecdf(x)))
            high_lim = x
            flag = True

    resolution = (high_lim-low_lim) 
    return resolution

def data_moving_average(data,start,end,window):
    """filter data using a moving average

    Args:
        data (pandas series): data to filter
        start (index): where to start in the dataset
        end (index): where to stop in the dataset
        window (int): size of data point window to use for moving average

    Returns:
        pandas series: data series with original data, filtered data and leftover noise data
    """    
    y = data.loc[start:end]
    y_filter = moving_average(data.loc[start-window:end+window],window)[window:-window]
    y_noise = y.copy(deep=True)    
    y_noise.iloc[:] = np.array(np.subtract(y.values,y_filter)).tolist()
    return y,y_filter,y_noise
