from time import sleep
import pandas as pd
import numpy as np
import math
import os
import seaborn as sb
import matplotlib.pyplot as plt
from statsmodels import stats
import statsmodels.api as sm
from scipy import stats as ss

# linear regression function definition
f_linreg = lambda a,b,x : a*x+b
# figure settings
steps = 10
time_divisor = 10**9 # ns to secs

def plot_dataset(dataset,filename,columns_y,column_x):
    """    opens a file in a folder, reads it according to predetermined structure, plot specified column names
    Args:
        dataset (Pandas Dataframe): data to plot
        filename (string): filename
        columns_y (string): column names for y data
        column_x (string): column name for x data
    """    
    print("---------------------PLOT DATASET---------------------------")
    for i in columns_y:
        x = dataset[column_x]
        y = dataset[i]
        fig, ax = plt.subplots(figsize=(15,7))
        sb.scatterplot(x=x, y=y, data=dataset,color='red').set(title=filename)
        xmin, xmax= min(x),max(x)
        step = round((xmax-xmin)/steps)
        ax.set_xticks(range(xmin,xmax,step))
        ax.set_xticklabels(range(xmin,xmax,step))

        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.savefig('Figures/'+filename[:-3]+'_'+i+'.png') # this overwrites existing files!

def read_file(folder,file,column_names,column_x):
    """    opens a file in a folder, reads it according to predetermined structure, return data (filename, data, statistics)
    Args:
        folder (string): folder where files are stored
        file (string): filename
        column_names (string): column names
        column_time (string): column name for x data

    Returns:
        [type]: [description]
    """
    fileName = str(file)
    print("---------------------READ FILE---------------------------")
    # first read the general setup settings
    f = open(folder+"\\"+fileName, "r")
    print("OPENED FILE: "+fileName)

    # read first line, check to see if it is the sensor number, if not keep on reading lines until sensor number encountered
    # next read all the settings, note that this order is defined by Arduino programming
    sensor = f.readline().split(",")[-1].split(" ")
    i = 1
    while sensor[1] != "SENSOR":
        sensor = f.readline().split(",")[-1].split(" ")
        i = i+1
        if i>100:
            print("no appropriate start found in first 100 lines")
            break
    sensor_number = int(sensor[-1])
    window_factor = float(f.readline().split(" ")[-1][:-1])
    sensitivity = f.readline().split(" ")[-1][:-1]
    target = float(f.readline().split(",")[-1][1:-1])
    threshold = float(f.readline().split(",")[-1][1:-1])
    startrow = i + 5
    settings = [sensor_number,window_factor,sensitivity,target,threshold]
    print("sensor: ", sensor_number,"| window factor: ",window_factor, "| sensitivity: " + sensitivity, "| target: ",target, "| threshold: ",threshold)
    f.close()

    # read data into dataframe to perform calculations
    # first i lines, and an extra one (5) can all be skipped by pandas (hence the i + 5)
    # different dataframes are stored in a dictionary with keys numbers according to read order
    data= pd.read_csv(folder+"\\"+fileName, skiprows = startrow, delimiter = ",")
    data.columns = column_names
    data[column_x] = (data[column_x].values/time_divisor).astype(int)
    # this changes the time to cumulative time, this is not needed in later datafiles
    # fgdos_data.iat[0,0] = 0
    # time = 0
    # for i in range(0,len(fgdos_data)):
    #     fgdos_data['time'][i] = fgdos_data['time'][i] + time
    #     time = fgdos_data['time'][i]

    # always close file or use with open() as: ... (this closes file at end of with automatically)
    return [fileName],data,settings

def plot_lin_reg(dataset,filename,column_y,column_x):
    """performs linear regression upon data and plots data, linreg and endpoints

    Args:
        dataset (Pandas dataframe): pandas dataset
        filename (string): name of file to read
        column_y (string): column name for y data
        column_x (string): column name for x data
    """
    print("---------------------LIN REG---------------------------")
    x = dataset[column_x]
    y = dataset[column_y]
    fig, ax = plt.subplots(figsize=(15,7))
    sb.regplot(x=x,y=y ,data=dataset,scatter=True,scatter_kws={'linewidth': 2}).set(title=filename)
    sb.lineplot(x=x, y=y, data=dataset,color='red')
    xmin, xmax= min(x),max(x)
    step = round((xmax-xmin)/steps)
    ax.set_xticks(range(xmin,xmax,step))
    ax.set_xticklabels(range(xmin,xmax,step))

    x = sm.add_constant(x, has_constant='add') # add constant to determine intercept
    mod = sm.OLS(exog=x,endog=y)
    res = mod.fit()
    print(res.summary())
    intercept, coeff = res.params
    t_min, t_max = min(x), max(x)
    f_start, f_end = min(y), max(y)
    f_start_linreg, f_end_linreg = f_linreg(coeff,intercept,t_min), f_linreg(coeff,intercept,t_max)

    print("max frequency: ", f_start," | min frequency: ", f_end)
    print("max lin reg: ", f_start_linreg, " | min lin reg: ", f_end_linreg)
    plt.scatter([t_min],[f_start_linreg],c="green",marker='x',s=70)
    plt.scatter([t_max],[f_end_linreg],c="green",marker='x',s=70)
    ax.annotate(f_start_linreg,(t_min,f_start_linreg))
    ax.annotate(f_end_linreg,(t_max,f_end_linreg))

    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.savefig('Figures/'+filename[:-3]+'_linreg.png') # this overwrites existing files!

def plot_stats_normal_student(dataset,filename,column_y):
    """
    plot histogram, normal distribution, student t distribution
    returns confidence interval for mean

    Args:
        dataset (Pandas Dataframe): data to analyse
        filename (string): filename
        column_frequency (string): column name for y data

    Returns:
        list: list of confidence intervals at different levels
    """
    
    print("---------------------STATS---------------------------")
    # normal distr (loc = mean, scale = std)
    # pandas factor for std is n-1, which can be changed in arguments with ddof
    indeces = [0,1,2,3,7]
    count,mean,std,min,max = [pd.DataFrame.describe(dataset[column_y])[x] for x in indeces]
    # print(pd.DataFrame.std(dataset[Fs],ddof=0))
    # print(ss.norm.fit(dataset[Fs]))
    print("pandas data stats | mean:  ",mean," | standard deviation: ",std," | count: ",count)
    x = np.linspace(mean-4*std,mean+4*std,1000)
    dist_norm = ss.norm.pdf(x,mean,std)
    fig, ax = plt.subplots()
    ax.plot(x,dist_norm,'g--')

    # histogram
    hist, bins = np.histogram(dataset[column_y],10)
    ax.hist(dataset[column_y],bins=bins,density=True,alpha=0.5)

    # # student t distribution plot as an exercise (useless statistically speaking) (change dof te see added uncertainty)
    # loc, scale = mean, std
    # # print("scale estimate: ",scale*math.sqrt((count-1)/(count))) # *math.sqrt((count-3)/(count-1))
    # loc,scale = [ss.t.fit(dataset[column_frequency])[x] for x in [1,2]]
    # print("student t fit loc: ",loc," | scale: ",scale)
    # ax.plot(x,ss.t.pdf(x,count-1,loc,scale),'r-',alpha=0.5)

    # 95% confidence interval for mean using t-dist, calculated in different ways to double check
    # print(ss.t.interval(0.95,count-1))
    # print(ss.t.ppf(0.975,count-1))
    conf_interval = ss.t.interval(0.95,count-1,loc=mean,scale=std/math.sqrt(count))
    print("mean confidence interval (95%): ",conf_interval)
    # int = [mean-ss.t.ppf(0.975,count-1)*std/math.sqrt(count) , mean+ss.t.ppf(0.975,count-1)*std/math.sqrt(count)]
    # print(int)

    # give time to draw plots and continue with program aftwerwards
    plt.grid()
    plt.title(filename)
    plt.suptitle('Histogram')
    plt.xlabel(column_y)
    plt.ylabel('Distribution')
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.savefig('Figures/'+filename[:-3]+'_histogram.png') # this overwrites existing files!

    return list(conf_interval)

def t_tests_2sample(dataset_1,dataset_2,column_y):
    """
    compare 2 means by means of 2 sample t test
    assume underlying data follows normal distribution
    test for same variance (if not, other test procedure, see eg https://www.marsja.se/how-to-perform-a-two-sample-t-test-with-python-3-different-methods/ )

    Args:
        dataset_1 (Pandas Dataframes): datasets to compare
        dataset_2 (Pandas Dataframes): datasets to compare
        column_y (string): column name for data to test

    Returns:
        float: p test value
    """
    print("---------------------T TEST---------------------------")
    values = [[]]*2
    statistics = [[]*5]*2
    i = 0
    for dataset in [dataset_1,dataset_2]:
        values[i] = dataset[column_y].values
        indeces = [0,1,2,3,7]
        count,mean,std,min,max = [pd.DataFrame.describe(dataset[column_y])[x] for x in indeces]
        statistics[i] = [count,mean,std,min,max]
        i += 1

    # use trimmed mean for datasets with heavy tails (like Cauchy, which is a special version of student t)
    if (ss.levene(values[0],values[1],center='trimmed')[1] > 0.05):
        print("LEVENE TEST PASSED")
        ptest = ss.ttest_ind(values[0], values[1], equal_var=True)
        pvalue = ptest[1]
        if pvalue > 0.05:
            print("p test supports HO (mu1 = mu2) with value: ",pvalue)
        else:
            print("p test does not support HO (mu1 = mu2) with value: ",pvalue)

    else:
        print("LEVENE FAILED, TRY WELCH TEST")
        ptest = ss.ttest_ind(values[0], values[1], equal_var=False)
        pvalue = ptest[1]
        if pvalue > 0.05:
            print("p test supports HO (mu1 = mu2) with value: ",pvalue)
        else:
            print("p test does not support HO (mu1 = mu2) with value: ",pvalue)

    return pvalue