import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats as ss

# variables for dataset column names
t = "Time (ms)"
shunt_info_1 = 'V_shunt_1'
v_shunt_1 = 'V_s1 (µV)'
bus_info_1 = 'V_bus_1'
v_bus_1 = 'V_b1 (mV)'
i_shunt_1 = 'I_s1 (µA)'
p_1 = 'P_1 (mW)'
shunt_info_2 = 'V_shunt_2'
v_shunt_2 = 'V_s2 (µV)'
bus_info_2 = 'V_bus_2'
v_bus_2 = 'V_b2 (mV)'
i_shunt_2 = 'I_s2 (µA)'
p_2 = 'P_2 (mW)'
time_divisor = 10**6 # ns to ms
R_shunt = 0.1 # ohm
# list of summary statistics for pandas dataframe
pd_stats_list = ["min","max","mean","median","std","var","skew"]

# set matplotlib label size
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'font.weight': 555})
plt.rcParams.update({'font.family': 'sans-serif'})

def modify_file(folder,file,lineskip):
    """ 
    opens a file in a folder, reads it according to predetermined structure, modifies it to delete unnecessary rows and add current and power, return all necessary data (filename, data)
    """
    print("---------------------READ & MODIFY FILE---------------------------",file)
    
    # take out invalid lines and diminish number for faster processing. Lines with only 9 or more elements are stored.
    # if corrected file already exists, overwrite. By setting counter the number of new lines can be reduced
    file_name = str(file)
    counter = 0
    newfilename = folder+'\\corrected\\'+file_name[:-4]+'_corrected.csv'
    with open(folder+"\\"+file_name, "r") as f:
        with open(newfilename,'w') as new:
            new.seek(0)
            for line in f:
                counter += 1
                split = line.split(',')
                if (len(split)==9 and counter > lineskip):
                    new.write(line)
                    new.truncate()
                    counter = 0
                if (len(split)>9 and counter > lineskip):
                    line = ','.join(split[0:9])+'\n'
                    new.write(line)
                    new.truncate()
                    counter = 0
                
    '''
    # read lines until V is encountered. Created new file in which lines with only 5 elements are stored.
    with open(newfilename,"r"):
        data = f.readline()
        i = 1
        while (not data.find('V')):
            data = f.readline()
            i +=1
    '''
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

def read_file(folder,file):
    """ 
    opens a file in a folder, reads it according to predetermined structure, return all necessary data (filename, data)
    """
    print("---------------------READ FILE---------------------------",file)
    file_name = str(file)
    file_folder = folder+'\\'+file_name

    # read data into dataframe to perform calculations
    data= pd.read_csv(file_folder, skiprows = 0, delimiter = ",")

    return [file_name],data


def plot_data(folder,dataset,filename,steps,columns):
    """plots the data given by the datafile

    Args:
        dataset (Pandas dataframe): appropriate naming of columns necessary
        filename (string): 
    """
    print("---------------------PLOT DATA---------------------------")
    x = dataset[columns[0]]
    y = dataset[columns[1]]
    z = dataset[columns[2]]

    fig, axs = plt.subplots(2,1,figsize=(15,7))
    sb.scatterplot(ax=axs[0],x=x, y=y, data=dataset,color='red')
    xmin, xmax= min(x),max(x)
    step = round((xmax-xmin)/steps)
    axs[0].set_xticks(range(xmin,xmax,step))
    axs[0].set_xticklabels(range(xmin,xmax,step))
    sb.scatterplot(ax=axs[1],x=x, y=z, data=dataset,color='blue')
    axs[1].set_xticks(range(xmin,xmax,step))
    axs[1].set_xticklabels(range(xmin,xmax,step))
    # fig.suptitle(filename)
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.savefig(folder+"\\"+str(filename)+'.png') # this overwrites existing files!

def hist_data(folder,dataset,filename,columns,dist='normal',bins=5):
    print("---------------------HIST DATA---------------------------")
    a = dataset[columns[0]]
    b = dataset[columns[1]]
    colours = ['indianred','royalblue']
    ranges = [0,0]

    fig, axs = plt.subplots(2,1,figsize=(15,7))
    i = 0
    for z in [a,b]:
        axs[i].hist(z,density=True,bins=bins,color=colours[i],alpha=0.6)
        axs[i].grid()
        axs[i].set_xlabel(columns[i])
        zmin, zmax= min(z),max(z)
        ranges[i] = zmax-zmin
        x = np.linspace(zmin, zmax, 500)
        if (str(dist)=='none'):
            pass
        elif (str(dist)=='normal'):
            mu, std = ss.norm.fit(z)
            p = ss.norm.pdf(x, mu, std)
            plt.plot(x, p, linewidth=2, color = 'royalblue',label = 'Normal Distribution')
            param_1,param_2 = mu, std
            dist = p
        elif (str(dist)=='kde'):
            kde = ss.gaussian_kde(z)
            plt.plot(x,kde.evaluate(x), color = 'royalblue', label = 'Gaussian KDE')
            param_1,param_2 = kde.factor, kde.covariance
            dist = kde
        i += 1 
    
    # fig.suptitle(filename)
    fig.tight_layout()
    fig.show()
    fig.savefig(folder+'\\'+filename+'.png')

    return ranges