from statistics import mean
from tkinter.tix import ListNoteBook
from FGD_03F_functions import *

# useful links noise
# https://bjpcjp.github.io/pdfs/cmos_layout_sim/ch08-noise.pdf
# https://www.allaboutcircuits.com/technical-articles/noise-in-electronics-engineering-distribution-noise-rms-peak-to-peak-value-PSD/
# https://www.allaboutcircuits.com/technical-articles/introduction-to-statistical-noise-analysis-basic-calculations/
# useful links Empirical Distribution Function
# https://machinelearningmastery.com/empirical-distribution-function-in-python/
# useful links Gaussian KDE
# https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html

def Noise_01_HIGH_Dec_15_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS Noise_01_HIGH_Dec_15 ********************************************")
    print('interesting: Noise with Arduino powered via USB')
    savefolder = os.getcwd()+'\\Figures\\Noise\\Noise_01_HIGH_Dec_15'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1[t_info], dataset_2[t_info] = dataset_1[t_info]/1000, dataset_2[t_info]/1000
    t_info_s = 'Time (s)'
    dataset_1.rename(columns={t_info : t_info_s}, inplace=True)
    dataset_2.rename(columns={t_info : t_info_s}, inplace=True)
    
    # plot temperature to see of compensation might be needed
    # plot reference frequency and sensor frequency to see variation over time
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info_s],dataset_2[t_info_s],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[80,87],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y) # limits are off due to error
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'],markers=['.','.'])
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    y = [y_1_s, y_2_s]
    names = ['F sensor 1','F sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'])

    # take 10 min, from 630 to 1000 s (why?)
    i_start_1, i_start_2 = dataset_1.index[dataset_1[t_info_s]>=630].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=630].tolist()[0]
    i_end_1, i_end_2 = dataset_1.index[dataset_1[t_info_s]>=1000].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=1000].tolist()[0]

    # Histogram normality test and 6*sigma noise range limits
    x_1_l, x_2_l = dataset_1.loc[i_start_1:i_end_1,[F_s_info]] , dataset_2.loc[i_start_2:i_end_2,[F_s_info]]
    print("number of data points S1: %d"%(len(x_1_l[F_s_info])))
    fig, ax, kde = hist_dist_plot(x_1_l,file_name_1[0:-7]+'_histogram_S1','Sensor 1', colors=['indianred','royalblue'],savefolder=savefolder, bins=18, dist = 'kde')[0:3]
    ecdf_1 = sm.distributions.ECDF(x_1_l[F_s_info])
    # ax2 = ax.twinx()
    # ax2.plot(ecdf_1.x,ecdf_1.y,label='ECDF',color='darkblue')
    # ax2.set_ylim([0,1])
    # ax2.legend()
    xmin, xmax = limits([x_1_l[F_s_info]])
    steps = 1000
    x = np.linspace(xmin, xmax, steps)
    flag = True
    for x in x:
        kde_int = kde.integrate_box_1d(xmin,x)
        if (kde_int>=0.013 and kde_int<=0.987 and flag):
            print("Gaussian KDE lower noise range limit (CDF = 0.013%%) (CDF | Hz) (S1): %f | %f "%(kde_int,x))
            print("ECDF double check: %f"%(ecdf_1(x)))
            low_lim = x
            flag = False
        if(kde_int>=0.987 and not flag):
            print("Gaussian KDE higher noise range limit (CDF = 0.987%%) (CDF | Hz) (S1): %f | %f "%(kde_int,x))
            print("ECDF double check: %f"%(ecdf_1(x)))
            high_lim = x
            flag = True
    sensitivity = 10 # kHz/Gy
    resolution_1 = (high_lim-low_lim)/(sensitivity)

    mu_2,std_2 = hist_dist_plot(x_2_l,file_name_1[0:-7]+'_histogram_S2','Sensor 2', colors=['royalblue','indianred'],savefolder=savefolder, bins = 20)[3:5]
    print("Noise peak to peak range limits (+- 3*sigma = 99.73%%) S2 (mu | 3s ): %f | %f "%(mu_2,3*std_2))
    sensitivity = 10 # kHz/Gy

    print("Resolution assuming %d (kHz/Gy) sensitivity (mGy) (S1 | S2): %.3f | %.3f "%(sensitivity,resolution_1,6*std_2/(sensitivity)))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def Noise_02_HIGH_Dec_15_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS Noise_02_HIGH_Dec_15 ********************************************")
    print('interesting: Noise, resolution, distribution')
    savefolder = os.getcwd()+'\\Figures\\Noise\\Noise_02_HIGH_Dec_15'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1[t_info], dataset_2[t_info] = dataset_1[t_info]/1000, dataset_2[t_info]/1000
    t_info_s = 'Time (s)'
    dataset_1.rename(columns={t_info : t_info_s}, inplace=True)
    dataset_2.rename(columns={t_info : t_info_s}, inplace=True)
    
    # plot temperature to see of compensation might be needed
    # plot reference frequency and sensor frequency to see variation over time
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info_s],dataset_2[t_info_s],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[80,87],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y) # limits are off due to error
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'],markers=['.','.'])
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    y = [y_1_s, y_2_s]
    names = ['F sensor 1','F sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'])

    # take 10 min, from 500 to 1100 s (why?)
    print("time franme for tests: %d | %d"%(500, 1100))
    i_start_1, i_start_2 = dataset_1.index[dataset_1[t_info_s]>=500].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=500].tolist()[0]
    i_end_1, i_end_2 = dataset_1.index[dataset_1[t_info_s]>=1100].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=1100].tolist()[0]

    # Histogram normality test and 4*sigma noise range limits
    x_1_l, x_2_l = dataset_1.loc[i_start_1:i_end_1,F_s_info] , dataset_2.loc[i_start_2:i_end_2,F_s_info]
    data_points_1, data_points_2 = len(x_1_l.index), len(x_2_l.index)
    print("data points (S1 | S2): %d | %d"%(data_points_1,data_points_2))
    mu_1,std_1 = hist_dist_plot(x_1_l,F_s_info,file_name_1[0:-7]+'_histogram_S1','Sensor 1', colors=['indianred','royalblue'],savefolder=savefolder, bins=25,dist='normal')[3:5]
    mu_2,std_2 = hist_dist_plot(x_2_l,F_s_info,file_name_1[0:-7]+'_histogram_S2','Sensor 2', colors=['royalblue','indianred'],savefolder=savefolder, bins = 25,dist='normal')[3:5]
    print("Noise peak to peak range limits (+- 2*sigma = 95.45%%) S1 (mu | 4s ): %f | %f "%(mu_1,4*std_1))
    print("Noise peak to peak range limits (+- 2*sigma = 95.45%%) S2 (mu | 4s ): %f | %f "%(mu_2,4*std_2))
    sensitivity = 10 # kHz/Gy
    print("Resolution assuming %d (kHz/Gy) sensitivity (mGy) (S1 | S2): %.3f | %.3f "%(sensitivity,4*std_1/(sensitivity),4*std_2/(sensitivity)))
    
    # Shapiro Wilk normality, but only works for continuous variables
    print('NOTE: shapiro wilk for continuous variables, so useless')
    W_1, p_1 = ss.shapiro(x_1_l)
    W_2, p_2 = ss.shapiro(x_2_l)
    print("Shapiro Wilk normality test (W,p) (S1 | S2): (%.4f,%.4f) | (%.4f,%.4f)"%(W_1,p_1,W_2,p_2))
    if (p_1>0.05 and p_2>0.05):
        print("SW normality test passed (p > 0.05, W ~ 1)")
    else:
        print("SW normality test failed")
        # print(ft.get_common_distributions())
        # print(ft.get_distributions())
        # fig, ax = plt.subplots(figsize=(15,7))
        # f = ft.Fitter(x_1_l,distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw', 'rayleigh', 'uniform'])
        # f.fit()
        # f.summary()
        # fig.show()

    # low pass Butterworth filter
    # x_1_f = dataset_1.loc[i_start_1:i_end_1,t_info_s].values
    # y_1_f = dataset_1.loc[i_start_1:i_end_1,F_s_info].values
    # fig,ax = plot_xy([x_1_f],[y_1_f],file_name_1[0:-7]+'_Filtering',names,steps,savefolder=savefolder,colors=['darkred','darkblue'])
    # fs = 1000  # Sampling frequency
    # t = np.array(x_1_f)
    # fc = 30  # Cut-off frequency of the filter
    # w = fc / (fs / 2) # Normalize the frequency
    # b, a = ssig.butter(5, w, 'low',analog=False)
    # output = ssig.filtfilt(b, a, y_1_f)
    # plt.plot(t, output, label='filtered')
    # plt.legend()
    # plt.show()

    # moving average filter as done by CERN, apply analysis to new data set
    x_1_new, x_2_new = dataset_1.loc[i_start_1:i_end_1,t_info_s], dataset_2.loc[i_start_2:i_end_2,t_info_s]
    window = 80 # 40 for about 10s moving average window
    t_window = 2*4096/31.25 # ms
    print('measurement window is (2*4096 pulses / 31.25 kHz clock) (ms): %f'%(t_window))
    print('moving average window (ms): %f'%(t_window*window))
    y_1_new, y_1_filter, y_1_noise = data_moving_average(dataset_1.loc[:,F_s_info],i_start_1,i_end_1,window)
    y_2_new, y_2_filter, y_2_noise = data_moving_average(dataset_2[F_s_info],i_start_2,i_end_2,window)

    fig,ax = plot_xy([x_1_new,x_1_new],[y_1_new,y_1_filter],file_name_1[0:-7]+'_Filtering_S1',['S1', 'S1 Filtered'],steps,savefolder=savefolder,colors=['darkred','darkblue'])
    fig,ax = plot_xy([x_1_new,],[y_1_noise],file_name_1[0:-7]+'_Filtered_S1',['S1 Filtered'],steps,savefolder=savefolder,colors=['darkred',])
    fig,ax = plot_xy([x_2_new,x_2_new],[y_2_new,y_2_filter],file_name_1[0:-7]+'_Filtering_S2',['S2', 'S2 Filtered'],steps,savefolder=savefolder,colors=['darkblue','darkred'])
    fig,ax = plot_xy([x_2_new,],[y_2_noise],file_name_1[0:-7]+'_Filtered_S2',['S2 Filtered'],steps,savefolder=savefolder,colors=['darkblue'])
    data_points_1, data_points_2 = len(y_1_noise.index),len(y_2_noise.index)
    print("data points (S1 | S2): %d | %d"%(data_points_1,data_points_2))
    mu_1,std_1 = hist_dist_plot(y_1_noise,F_s_info,file_name_1[0:-7]+'_histogram_S1_noise','Sensor 1', colors=['indianred','royalblue'],savefolder=savefolder, bins=25,dist='normal')[3:5]
    mu_2,std_2 = hist_dist_plot(y_2_noise,F_s_info,file_name_1[0:-7]+'_histogram_S2_noise','Sensor 2', colors=['royalblue','indianred'],savefolder=savefolder, bins = 25,dist='normal')[3:5]
    print("Noise peak to peak range limits (+- 2*sigma = 95.45%%) S1 (mu | 4s ): %f | %f "%(mu_1,4*std_1))
    print("Noise peak to peak range limits (+- 2*sigma = 95.45%%) S2 (mu | 4s ): %f | %f "%(mu_2,4*std_2))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def Noise_03_LOW_Dec_15_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS Noise_03_LOW_Dec_15 ********************************************")
    print('interesting: Noise, bimodal distribution, resolution, temperature')
    savefolder = os.getcwd()+'\\Figures\\Noise\\Noise_03_LOW_Dec_15'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1[t_info], dataset_2[t_info] = dataset_1[t_info]/1000, dataset_2[t_info]/1000
    t_info_s = 'Time (s)'
    dataset_1.rename(columns={t_info : t_info_s}, inplace=True)
    dataset_2.rename(columns={t_info : t_info_s}, inplace=True)
    
    # plot temperature to see of compensation might be needed
    # plot reference frequency and sensor frequency to see variation over time
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info_s],dataset_2[t_info_s],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[80,87],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y) # limits are off due to error
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'],markers=['.','.'])
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    y = [y_1_s, y_2_s]
    names = ['F sensor 1','F sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'])

    # take 10 min, from 500 to 1100 s (why?)
    i_start_1, i_start_2 = dataset_1.index[dataset_1[t_info_s]>=500].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=500].tolist()[0]
    i_end_1, i_end_2 = dataset_1.index[dataset_1[t_info_s]>=1100].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=1100].tolist()[0]

    # Histogram distribution plot and 6*sigma (99.73% range limits for other distributions) noise range limits
    # Use gaussian ECDF to double check values ()
    x_1_l, x_2_l = dataset_1.loc[i_start_1:i_end_1,F_s_info] , dataset_2.loc[i_start_2:i_end_2,F_s_info]
    print("number of data points S1: %d"%(len(x_1_l[F_s_info])))
    fig, ax, kde = hist_dist_plot(x_1_l,F_s_info,file_name_1[0:-7]+'_histogram_S1','Sensor 1', colors=['indianred','royalblue'],savefolder=savefolder, bins=31, dist = 'kde')[0:3]
    ecdf_1 = sm.distributions.ECDF(x_1_l)
    # ax2 = ax.twinx()
    # ax2.plot(ecdf_1.x,ecdf_1.y,label='ECDF',color='darkblue')
    # ax2.set_ylim([0,1])
    # ax2.legend()
    xmin, xmax = limits([x_1_l[F_s_info]])
    steps = 1000
    x = np.linspace(xmin, xmax, steps)
    flag = True
    for x in x:
        kde_int = kde.integrate_box_1d(xmin,x)
        if (kde_int>=0.013 and kde_int<=0.987 and flag):
            print("Gaussian KDE lower noise range limit (CDF = 0.013%%) (CDF | Hz) (S1): %f | %f "%(kde_int,x))
            print("ECDF double check: %f"%(ecdf_1(x)))
            low_lim = x
            flag = False
        if(kde_int>=0.987 and not flag):
            print("Gaussian KDE higher noise range limit (CDF = 0.987%%) (CDF | Hz) (S1): %f | %f "%(kde_int,x))
            print("ECDF double check: %f"%(ecdf_1(x)))
            high_lim = x
            flag = True
    sensitivity = 1.5 # kHz/Gy
    resolution_1 = (high_lim-low_lim)/(sensitivity)

    fig, ax, kde = hist_dist_plot(x_2_l,F_s_info,file_name_1[0:-7]+'_histogram_S2','Sensor 2', colors=['royalblue','indianred'],savefolder=savefolder, bins=27, dist = 'kde')[0:3]
    ecdf_2 = sm.distributions.ECDF(x_2_l)
    # ax2 = ax.twinx()
    # ax2.plot(ecdf_2.x,ecdf_2.y,label='ECDF',color='darkred')
    # ax2.set_ylim([0,1])
    # ax2.legend()
    xmin, xmax = limits([x_2_l])
    steps = 1000
    x = np.linspace(xmin, xmax, steps)
    flag = True
    for x in x:
        kde_int = kde.integrate_box_1d(xmin,x)
        if (kde_int>=0.013 and kde_int<=0.987 and flag):
            print("Gaussian KDE lower noise range limit (CDF = 0.013%%) (CDF | Hz) (S2): %f | %f "%(kde_int,x))
            print("ECDF double check: %f"%(ecdf_2(x)))
            low_lim = x
            flag = False
        if(kde_int>=0.987 and not flag):
            print("Gaussian KDE higher noise range limit (CDF = 0.987%%) (CDF | Hz) (S2): %f | %f "%(kde_int,x))
            print("ECDF double check: %f"%(ecdf_2(x)))
            high_lim = x
            flag = True
    sensitivity = 1.5 # kHz/Gy
    resolution_2 = (high_lim-low_lim)/(sensitivity)
    
    print("Resolution assuming %f (kHz/Gy) sensitivity (mGy) (S1 | S2): %.3f | %.3f "%(sensitivity,resolution_1,resolution_2))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def Noise_04_HIGH_Dec_16_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS Noise_04_HIGH_Dec_16 ********************************************")
    print('interesting: Noise temperature influence')
    savefolder = os.getcwd()+'\\Figures\\Noise\\Noise_04_HIGH_Dec_16'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1[t_info], dataset_2[t_info] = dataset_1[t_info]/1000, dataset_2[t_info]/1000
    t_info_s = 'Time (s)'
    dataset_1.rename(columns={t_info : t_info_s}, inplace=True)
    dataset_2.rename(columns={t_info : t_info_s}, inplace=True)
    
    # plot temperature to see of compensation might be needed
    # plot reference frequency and sensor frequency to see variation over time
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info_s],dataset_2[t_info_s],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[77,87],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y) # limits are off due to error
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'],markers=['.','.'])
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    y = [y_1_s, y_2_s]
    names = ['F sensor 1','F sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'])

    # # take 10 min, from 500 to 1100 s (why?)
    # i_start_1, i_start_2 = dataset_1.index[dataset_1[t_info_s]>=50].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=50].tolist()[0]
    # i_end_1, i_end_2 = dataset_1.index[dataset_1[t_info_s]>=600].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=600].tolist()[0]

    # # Histogram normality test and 6*sigma noise range limits
    # x_1_l, x_2_l = dataset_1.loc[i_start_1:i_end_1,[F_s_info]] , dataset_2.loc[i_start_2:i_end_2,[F_s_info]]
    # hist_dist_plot(x_1_l,file_name_1[0:-7]+'_histogram_S1','Sensor 1', colors=['indianred','royalblue'],savefolder=savefolder, bins=25,dist='kde')
    # hist_dist_plot(x_2_l,file_name_1[0:-7]+'_histogram_S2','Sensor 2', colors=['royalblue','indianred'],savefolder=savefolder, bins = 25,dist='kde')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def HIGH_Jan_14_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS HIGH_Jan_14 ********************************************")
    print('interesting: Noise temperature influence')
    savefolder = os.getcwd()+'\\Figures\\Noise\\HIGH_Jan_14'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1[t_info], dataset_2[t_info] = dataset_1[t_info]/1000, dataset_2[t_info]/1000
    t_info_s = 'Time (s)'
    dataset_1.rename(columns={t_info : t_info_s}, inplace=True)
    dataset_2.rename(columns={t_info : t_info_s}, inplace=True)
    
    # plot temperature to see of compensation might be needed
    # plot reference frequency and sensor frequency to see variation over time
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info_s],dataset_2[t_info_s],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[77,87],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y) # limits are off due to error
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'],markers=['.','.'])
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    y = [y_1_s, y_2_s]
    names = ['F sensor 1','F sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'])

    # determine average over first 2 min
    F_start_1 = dataset_1.loc[dataset_1[t_info_s]<120,F_s_info].mean()
    F_start_2 = dataset_2.loc[dataset_2[t_info_s]<120,F_s_info].mean()
    print("start frequencies (mean over 120 s) (S1 | S2 ): %f | %f"%(F_start_1,F_start_2))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def HIGH_Jan_15_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS HIGH_Jan_15 ********************************************")
    print('interesting: Noise temperature influence')
    savefolder = os.getcwd()+'\\Figures\\Noise\\HIGH_Jan_15'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1[t_info], dataset_2[t_info] = dataset_1[t_info]/1000, dataset_2[t_info]/1000
    t_info_s = 'Time (s)'
    dataset_1.rename(columns={t_info : t_info_s}, inplace=True)
    dataset_2.rename(columns={t_info : t_info_s}, inplace=True)
    
    # plot temperature to see of compensation might be needed
    # plot reference frequency and sensor frequency to see variation over time
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info_s],dataset_2[t_info_s],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[77,87],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y) # limits are off due to error
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'],markers=['.','.'])
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    y = [y_1_s, y_2_s]
    names = ['F sensor 1','F sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'])

    # take full range and compare to part of range! (discard first lower temperature points)
    print("OVER WHOLE RANGE")
    i_start_1, i_start_2 = dataset_1.index.tolist()[50] , dataset_2.index.tolist()[50]
    i_end_1, i_end_2 = dataset_1.index.tolist()[-1] , dataset_2.index.tolist()[-1]

    # Histogram and noise range limits
    x_1_l, x_2_l = dataset_1.loc[i_start_1:i_end_1,[F_s_info]] , dataset_2.loc[i_start_2:i_end_2,[F_s_info]]
    sensitivity = 10 # kHz/Gy
    print("number of data points S1: %d"%(len(x_1_l[F_s_info])))
    fig, ax, kde = hist_dist_plot(x_1_l,file_name_1[0:-7]+'_histogram_S1_all','Sensor 1', colors=['indianred','royalblue'],savefolder=savefolder, bins=29, dist = 'kde')[0:3]
    resolution_1 = kde_limits(x_1_l,1000,kde,sensitivity)
    print("number of data points S2: %d"%(len(x_2_l[F_s_info])))
    fig, ax, kde = hist_dist_plot(x_2_l,file_name_1[0:-7]+'_histogram_S2_all','Sensor 2', colors=['royalblue','indianred'],savefolder=savefolder, bins = 23,dist='kde')[0:3]
    resolution_2 = kde_limits(x_2_l,1000,kde,sensitivity)
    print("Resolution assuming %f (kHz/Gy) sensitivity (mGy) (S1 | S2): %.3f | %.3f "%(sensitivity,resolution_1,resolution_2))

    # take full range and compare to part of range!
    print("OVER PART RANGE (1250-1850")
    i_start_1, i_start_2 = dataset_1.index[dataset_1[t_info_s]>=1250].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=1250].tolist()[0]
    i_end_1, i_end_2 = dataset_1.index[dataset_1[t_info_s]>=1850].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=1850].tolist()[0]

    # Histogram and noise range limits
    x_1_l, x_2_l = dataset_1.loc[i_start_1:i_end_1,[F_s_info]] , dataset_2.loc[i_start_2:i_end_2,[F_s_info]]
    sensitivity = 10 # kHz/Gy
    print("number of data points S1: %d"%(len(x_1_l[F_s_info])))
    fig, ax, kde = hist_dist_plot(x_1_l,file_name_1[0:-7]+'_histogram_S1_part','Sensor 1', colors=['indianred','royalblue'],savefolder=savefolder, bins=23, dist = 'kde')[0:3]
    resolution_1 = kde_limits(x_1_l,1000,kde,sensitivity)
    print("number of data points S2: %d"%(len(x_2_l[F_s_info])))
    fig, ax, kde = hist_dist_plot(x_2_l,file_name_1[0:-7]+'_histogram_S2_part','Sensor 2', colors=['royalblue','indianred'],savefolder=savefolder, bins = 23,dist='kde')[0:3]
    resolution_2 = kde_limits(x_2_l,1000,kde,sensitivity)
    print("Resolution assuming %f (kHz/Gy) sensitivity (mGy) (S1 | S2): %.3f | %.3f "%(sensitivity,resolution_1,resolution_2))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def LOW_Jan_15_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS LOW_Jan_15 ********************************************")
    print('interesting: Noise temperature influence')
    savefolder = os.getcwd()+'\\Figures\\Noise\\LOW_Jan_15'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1[t_info], dataset_2[t_info] = dataset_1[t_info]/1000, dataset_2[t_info]/1000
    t_info_s = 'Time (s)'
    dataset_1.rename(columns={t_info : t_info_s}, inplace=True)
    dataset_2.rename(columns={t_info : t_info_s}, inplace=True)
    
    # plot temperature to see of compensation might be needed
    # plot reference frequency and sensor frequency to see variation over time
    steps = 30
    i_start,i_end = 1000, 4000
    # select only to see temperature influence
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info_s],dataset_2[t_info_s],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[77,87],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y) # limits are off due to error
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'],markers=['o','o'])
    y_1_s, y_2_s = dataset_1[F_s_info],dataset_2[F_s_info]
    y = [y_1_s, y_2_s]
    names = ['F sensor 1','F sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'])

    #  show that sensor has to settle in the beginning
    i_start_1, i_start_2 = dataset_1.index[dataset_1[t_info_s]>=0].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=0].tolist()[0]
    i_end_1, i_end_2 = dataset_1.index[dataset_1[t_info_s]>=600].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=600].tolist()[0]
    x_1, x_2 = dataset_1.loc[i_start_1:i_end_1,t_info_s],dataset_2.loc[i_start_2:i_end_2,t_info_s]
    y_1_s, y_2_s = dataset_1.loc[i_start_1:i_end_1,F_s_info],dataset_2.loc[i_start_2:i_end_2,F_s_info]
    names = ['F sensor 1','F sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs_settling',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'])

    #  show constant part (from 600 to 2400)
    print("time franme for tests: %d | %d"%(600, 2400))
    i_start_1, i_start_2 = dataset_1.index[dataset_1[t_info_s]>=600].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=600].tolist()[0]
    i_end_1, i_end_2 = dataset_1.index[dataset_1[t_info_s]>=2400].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=2400].tolist()[0]
    x_1, x_2 = dataset_1.loc[i_start_1:i_end_1,t_info_s],dataset_2.loc[i_start_2:i_end_2,t_info_s]
    y_1_s, y_2_s = dataset_1.loc[i_start_1:i_end_1,F_s_info],dataset_2.loc[i_start_2:i_end_2,F_s_info]
    names = ['F sensor 1','F sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs_constant',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'])

    # Histogram and noise range limits
    print("PART OF RANGE (600-2400)")
    x_1_l, x_2_l = x_1_l, x_2_l = dataset_1.loc[i_start_1:i_end_1,F_s_info] , dataset_2.loc[i_start_2:i_end_2,F_s_info]
    sensitivity = 10 # kHz/Gy
    print("number of data points S1: %d"%(len(x_1_l)))
    fig, ax, kde = hist_dist_plot(x_1_l,F_s_info,file_name_1[0:-7]+'_histogram_S1_all','Sensor 1', colors=['indianred','royalblue'],savefolder=savefolder, bins=28, dist = 'kde')[0:3]
    resolution_1 = kde_limits(x_1_l,5000,kde,confidence=0.9545)
    print("number of data points S2: %d"%(len(x_2_l)))
    fig, ax, kde = hist_dist_plot(x_2_l,F_s_info,file_name_1[0:-7]+'_histogram_S2_all','Sensor 2', colors=['royalblue','indianred'],savefolder=savefolder, bins = 28,dist='kde')[0:3]
    resolution_2 = kde_limits(x_2_l,5000,kde,confidence=0.9545)
    print("Resolution (Hz) (S1 | S2): %.3f | %.3f "%(resolution_1,resolution_2))
    mean_1, mean_2, variance_1, variance_2 = np.mean(x_1_l), np.mean(x_2_l), np.var(x_1_l), np.var(x_2_l)
    print("mean and variance (S1 | S2): %f , %f | %f , %f"%(mean_1,variance_1,mean_2,variance_2))

    # show temperature change part
    # dataset_1_T, dataset_2_T = dataset_1[dataset_1[t_info_s]>=2500], dataset_2[dataset_2[t_info_s]>=2500]
    # x_1_T,x_2_T,y_1_T,y_2_T = dataset_1_T[t_info_s],dataset_2_T[t_info_s],dataset_1_T[T_info],dataset_2_T[T_info]
    # x = [x_1_T,x_2_T]
    # y = [y_1_T,y_2_T]
    # xmin, xmax = limits(x)
    # ymin, ymax = limits(y)
    # names = ['T sensor 1','T sensor 2']
    # fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature_T',names,steps,savefolder=savefolder,xlim=[xmin,xmax],ylim=[77,87],colors=['darkred','darkblue'])
    # y_1_r,y_2_r = dataset_1_T[F_r_info],dataset_2_T[F_r_info]
    # y = [y_1_r,y_2_r]
    # ymin, ymax = limits(y) # limits are off due to error
    # names = ['F reference 1','F reference 2']
    # fig,ax = plot_2subs_datasets(x_1_T,x_2_T,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr_T',names,steps,savefolder=savefolder,colors=['darkred','darkblue'],markers=['o','o'])
    # y_1_s, y_2_s = dataset_1_T[F_s_info],dataset_2_T[F_s_info]
    # y = [y_1_s, y_2_s]
    # names = ['F sensor 1','F sensor 2']
    # fig,ax = plot_2subs_datasets(x_1_T,x_2_T,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs_T',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['o','o'])

    # moving average filter as done by CERN, apply analysis to new data set
    x_1_new, x_2_new = dataset_1.loc[i_start_1:i_end_1,t_info_s], dataset_2.loc[i_start_2:i_end_2,t_info_s]
    window = 80 # 40 for about 10s moving average window
    t_window = 2*4096/31.25 # ms
    print('measurement window is (2*4096 pulses / 31.25 kHz clock) (ms): %f'%(t_window))
    print('moving average window (ms): %f'%(t_window*window))
    y_1_new, y_1_filter, y_1_noise = data_moving_average(dataset_1.loc[:,F_s_info],i_start_1,i_end_1,window)
    y_2_new, y_2_filter, y_2_noise = data_moving_average(dataset_2[F_s_info],i_start_2,i_end_2,window)

    fig,ax = plot_xy([x_1_new,x_1_new],[y_1_new,y_1_filter],file_name_1[0:-7]+'_Filtering_S1',['S1', 'S1 Filtered'],steps,savefolder=savefolder,colors=['darkred','darkblue'])
    fig,ax = plot_xy([x_1_new,],[y_1_noise],file_name_1[0:-7]+'_Filtered_S1',['S1 Filtered'],steps,savefolder=savefolder,colors=['darkred',])
    fig,ax = plot_xy([x_2_new,x_2_new],[y_2_new,y_2_filter],file_name_1[0:-7]+'_Filtering_S2',['S2', 'S2 Filtered'],steps,savefolder=savefolder,colors=['darkblue','darkred'])
    fig,ax = plot_xy([x_2_new,],[y_2_noise],file_name_1[0:-7]+'_Filtered_S2',['S2 Filtered'],steps,savefolder=savefolder,colors=['darkblue'])
    data_points_1, data_points_2 = len(y_1_noise.index),len(y_2_noise.index)
    print("data points (S1 | S2): %d | %d"%(data_points_1,data_points_2))
    fig, ax, kde = hist_dist_plot(y_1_noise,F_s_info,file_name_1[0:-7]+'_histogram_S1_noise','Sensor 1', colors=['indianred','royalblue'],savefolder=savefolder, bins=28, dist = 'kde')[0:3]
    resolution_1 = kde_limits(y_1_noise,5000,kde,confidence=0.9545)
    fig, ax, kde = hist_dist_plot(y_2_noise,F_s_info,file_name_1[0:-7]+'_histogram_S2_noise','Sensor 2', colors=['royalblue','indianred'],savefolder=savefolder, bins = 28,dist='kde')[0:3]
    resolution_2 = kde_limits(y_2_noise,5000,kde,confidence=0.9545)
    print("Resolution (Hz) (S1 | S2): %.3f | %.3f "%(resolution_1,resolution_2))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def HIGH_Jan_26_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS HIGH_Jan_26 ********************************************")
    print('interesting: Noise temperature influence')
    savefolder = os.getcwd()+'\\Figures\\Noise\\HIGH_Jan_26'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1[t_info], dataset_2[t_info] = dataset_1[t_info]/1000, dataset_2[t_info]/1000
    t_info_s = 'Time (s)'
    dataset_1.rename(columns={t_info : t_info_s}, inplace=True)
    dataset_2.rename(columns={t_info : t_info_s}, inplace=True)
    
    # plot temperature to see of compensation might be needed
    # plot reference frequency and sensor frequency to see variation over time
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info_s],dataset_2[t_info_s],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[77,87],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y) # limits are off due to error
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'],markers=['.','.'])
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    y = [y_1_s, y_2_s]
    names = ['F sensor 1','F sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def HIGH_Feb_18_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS HIGH_Feb_18 ********************************************")
    print('interesting: ?')
    savefolder = os.getcwd()+'\\Figures\\Noise\\HIGH_Feb_18'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1[t_info], dataset_2[t_info] = dataset_1[t_info]/1000, dataset_2[t_info]/1000
    t_info_s = 'Time (s)'
    dataset_1.rename(columns={t_info : t_info_s}, inplace=True)
    dataset_2.rename(columns={t_info : t_info_s}, inplace=True)
    
    # plot temperature to see of compensation might be needed
    # plot reference frequency and sensor frequency to see variation over time
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info_s],dataset_2[t_info_s],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[77,87],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y) # limits are off due to error
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'],markers=['.','.'])
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    y = [y_1_s, y_2_s]
    names = ['F sensor 1','F sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def HIGH_w32768_Feb_22_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS HIGH_w32768_Feb_22 ********************************************")
    print('interesting: higher window pulses (32768)')
    savefolder = os.getcwd()+'\\Figures\\Noise\\HIGH_w32768_Feb_22'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1[t_info], dataset_2[t_info] = dataset_1[t_info]/1000, dataset_2[t_info]/1000
    t_info_s = 'Time (s)'
    dataset_1.rename(columns={t_info : t_info_s}, inplace=True)
    dataset_2.rename(columns={t_info : t_info_s}, inplace=True)
    
    # plot temperature to see of compensation might be needed
    # plot reference frequency and sensor frequency to see variation over time
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info_s],dataset_2[t_info_s],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[77,87],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y) # limits are off due to error
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'],markers=['.','.'])
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    y = [y_1_s, y_2_s]
    names = ['F sensor 1','F sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'],ylim_2=[131500,132200])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def HIGH_Feb_23_analysis(file_1,file_2,folder):

    print("******************************************** ANALYSIS HIGH_Feb_23 ********************************************")
    print('interesting: ?')
    savefolder = os.getcwd()+'\\Figures\\Noise\\HIGH_Feb_23'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1[t_info], dataset_2[t_info] = dataset_1[t_info]/1000, dataset_2[t_info]/1000
    t_info_s = 'Time (s)'
    dataset_1.rename(columns={t_info : t_info_s}, inplace=True)
    dataset_2.rename(columns={t_info : t_info_s}, inplace=True)
    
    # plot temperature to see of compensation might be needed
    # plot reference frequency and sensor frequency to see variation over time
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info_s],dataset_2[t_info_s],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[77,87],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y) # limits are off due to error
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'],markers=['.','.'])
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    y = [y_1_s, y_2_s]
    names = ['F sensor 1','F sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def HIGH_w32768_Feb_23_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS HIGH_w32768_Feb_23 ********************************************")
    print('interesting: higher window pulses (32768)')
    savefolder = os.getcwd()+'\\Figures\\Noise\\HIGH_w32768_Feb_23'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1[t_info], dataset_2[t_info] = dataset_1[t_info]/1000, dataset_2[t_info]/1000
    t_info_s = 'Time (s)'
    dataset_1.rename(columns={t_info : t_info_s}, inplace=True)
    dataset_2.rename(columns={t_info : t_info_s}, inplace=True)
    steps = 30

    # # plot temperature to see of compensation might be needed
    # # plot reference frequency and sensor frequency to see variation over time
    # x_1,x_2,y_1_T,y_2_T = dataset_1[t_info_s],dataset_2[t_info_s],dataset_1[T_info],dataset_2[T_info]
    # x = [x_1,x_2]
    # y = [y_1_T,y_2_T]
    # xmin, xmax = limits(x)
    # ymin, ymax = limits(y)
    # names = ['T sensor 1','T sensor 2']
    # fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[77,87],colors=['darkred','darkblue'])
    # y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    # y = [y_1_r,y_2_r]
    # ymin, ymax = limits(y) # limits are off due to error
    # names = ['F reference 1','F reference 2']
    # fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'],markers=['.','.'])
    # y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    # y = [y_1_s, y_2_s]
    # names = ['F sensor 1','F sensor 2']
    # fig,ax = plot_2subs_datasets(x_1,x_2,y_1_s,y_2_s,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,colors=['red','blue'],markers=['.','.'],ylim_2=[130800,131100])

    # Histogram and noise range limits (620 - 1220)
    i_start_1, i_start_2 = dataset_1.index[dataset_1[t_info_s]>=620].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=620].tolist()[0]
    i_end_1, i_end_2 = dataset_1.index[dataset_1[t_info_s]>=1220].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=1220].tolist()[0]
    print("PART OF RANGE (600-2400)")
    x_1_l, x_2_l = x_1_l, x_2_l = dataset_1.loc[i_start_1:i_end_1,F_s_info] , dataset_2.loc[i_start_2:i_end_2,F_s_info]
    print("number of data points (S1 | S2): %d | %d"%(len(x_1_l),len(x_2_l)))
    mu_1,std_1 = hist_dist_plot(x_1_l,F_s_info,file_name_1[0:-7]+'_histogram_S1','Sensor 1', colors=['indianred','royalblue'],savefolder=savefolder, bins=18,dist='normal')[3:5]
    mu_2,std_2 = hist_dist_plot(x_2_l,F_s_info,file_name_1[0:-7]+'_histogram_S2','Sensor 2', colors=['royalblue','indianred'],savefolder=savefolder, bins = 18,dist='normal')[3:5]
    print("Noise peak to peak range limits (+- 2*sigma = 95.45%%) S1 (mu | 4s ): %f | %f "%(mu_1,4*std_1))
    print("Noise peak to peak range limits (+- 2*sigma = 95.45%%) S2 (mu | 4s ): %f | %f "%(mu_2,4*std_2))

    # moving average filter as done by CERN, apply analysis to new data set (data BEFORE external arduino was unplugged)
    print("moving average data")
    x_1_new, x_2_new = dataset_1.loc[i_start_1:i_end_1,t_info_s], dataset_2.loc[i_start_2:i_end_2,t_info_s]
    window = 10 # 40 for about 10s moving average window
    t_window = 2*32768/31.25 # ms
    print('measurement window is (2*32768 pulses / 31.25 kHz clock) (ms): %f'%(t_window))
    print('moving average window (ms): %f'%(t_window*window))
    y_1_new, y_1_filter, y_1_noise = data_moving_average(dataset_1.loc[:,F_s_info],i_start_1,i_end_1,window)
    y_2_new, y_2_filter, y_2_noise = data_moving_average(dataset_2[F_s_info],i_start_2,i_end_2,window)
    fig,ax = plot_xy([x_1_new,x_1_new],[y_1_new,y_1_filter],file_name_1[0:-7]+'_Filtering_S1',['S1', 'S1 Filtered'],steps,savefolder=savefolder,colors=['darkred','darkblue'])
    fig,ax = plot_xy([x_1_new,],[y_1_noise],file_name_1[0:-7]+'_Filtered_S1',['S1 Filtered'],steps,savefolder=savefolder,colors=['darkred',])
    fig,ax = plot_xy([x_2_new,x_2_new],[y_2_new,y_2_filter],file_name_1[0:-7]+'_Filtering_S2',['S2', 'S2 Filtered'],steps,savefolder=savefolder,colors=['darkblue','darkred'])
    fig,ax = plot_xy([x_2_new,],[y_2_noise],file_name_1[0:-7]+'_Filtered_S2',['S2 Filtered'],steps,savefolder=savefolder,colors=['darkblue'])
    data_points_1, data_points_2 = len(y_1_noise.index),len(y_2_noise.index)
    print("data points (S1 | S2): %d | %d"%(data_points_1,data_points_2))
    mu_1,std_1 = hist_dist_plot(y_1_noise,F_s_info,file_name_1[0:-7]+'_histogram_S1_noise','Sensor 1', colors=['indianred','royalblue'],savefolder=savefolder, bins=18,dist='normal')[3:5]
    mu_2,std_2 = hist_dist_plot(y_2_noise,F_s_info,file_name_1[0:-7]+'_histogram_S2_noise','Sensor 2', colors=['royalblue','indianred'],savefolder=savefolder, bins = 18,dist='normal')[3:5]
    print("Noise peak to peak range limits (+- 2*sigma = 95.45%%) S1 (mu | 4s ): %f | %f "%(mu_1,4*std_1))
    print("Noise peak to peak range limits (+- 2*sigma = 95.45%%) S2 (mu | 4s ): %f | %f "%(mu_2,4*std_2))

    # EXTERNAL ARDUINO POWER UNPLUGGED, DO TESTS AGAIN TO SEE DIFFERENCE
    print("\n************** POWER UNPLUGGED DATA **************")
    # Histogram and noise range limits (2400 - 3000)
    i_start_1, i_start_2 = dataset_1.index[dataset_1[t_info_s]>=2400].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=2400].tolist()[0]
    i_end_1, i_end_2 = dataset_1.index[dataset_1[t_info_s]>=3000].tolist()[0] , dataset_2.index[dataset_2[t_info_s]>=3000].tolist()[0]
    print("PART OF RANGE (2400 - 3000)")
    x_1_l, x_2_l = x_1_l, x_2_l = dataset_1.loc[i_start_1:i_end_1,F_s_info] , dataset_2.loc[i_start_2:i_end_2,F_s_info]
    print("number of data points (S1 | S2): %d | %d"%(len(x_1_l),len(x_2_l)))
    mu_1,std_1 = hist_dist_plot(x_1_l,F_s_info,file_name_1[0:-7]+'_histogram_S1_power','Sensor 1', colors=['indianred','royalblue'],savefolder=savefolder, bins=18,dist='normal')[3:5]
    mu_2,std_2 = hist_dist_plot(x_2_l,F_s_info,file_name_1[0:-7]+'_histogram_S2_power','Sensor 2', colors=['royalblue','indianred'],savefolder=savefolder, bins = 18,dist='normal')[3:5]
    print("Noise peak to peak range limits (+- 2*sigma = 95.45%%) S1 (mu | 4s ): %f | %f "%(mu_1,4*std_1))
    print("Noise peak to peak range limits (+- 2*sigma = 95.45%%) S2 (mu | 4s ): %f | %f "%(mu_2,4*std_2))   

    # moving average filter as done by CERN, apply analysis to new data set (data AFTER external arduino was unplugged)
    print("moving average data")
    x_1_new, x_2_new = dataset_1.loc[i_start_1:i_end_1,t_info_s], dataset_2.loc[i_start_2:i_end_2,t_info_s]
    window = 10 # 40 for about 10s moving average window
    y_1_new, y_1_filter, y_1_noise = data_moving_average(dataset_1.loc[:,F_s_info],i_start_1,i_end_1,window)
    y_2_new, y_2_filter, y_2_noise = data_moving_average(dataset_2[F_s_info],i_start_2,i_end_2,window)
    fig,ax = plot_xy([x_1_new,x_1_new],[y_1_new,y_1_filter],file_name_1[0:-7]+'_Filtering_S1_power',['S1', 'S1 Filtered'],steps,savefolder=savefolder,colors=['darkred','darkblue'])
    fig,ax = plot_xy([x_1_new,],[y_1_noise],file_name_1[0:-7]+'_Filtered_S1_power',['S1 Filtered'],steps,savefolder=savefolder,colors=['darkred',])
    fig,ax = plot_xy([x_2_new,x_2_new],[y_2_new,y_2_filter],file_name_1[0:-7]+'_Filtering_S2_power',['S2', 'S2 Filtered'],steps,savefolder=savefolder,colors=['darkblue','darkred'])
    fig,ax = plot_xy([x_2_new,],[y_2_noise],file_name_1[0:-7]+'_Filtered_S2_power',['S2 Filtered'],steps,savefolder=savefolder,colors=['darkblue'])
    data_points_1, data_points_2 = len(y_1_noise.index),len(y_2_noise.index)
    print("data points (S1 | S2): %d | %d"%(data_points_1,data_points_2))
    mu_1,std_1 = hist_dist_plot(y_1_noise,F_s_info,file_name_1[0:-7]+'_histogram_S1_noise_power','Sensor 1', colors=['indianred','royalblue'],savefolder=savefolder, bins=18,dist='normal')[3:5]
    mu_2,std_2 = hist_dist_plot(y_2_noise,F_s_info,file_name_1[0:-7]+'_histogram_S2_noise_power','Sensor 2', colors=['royalblue','indianred'],savefolder=savefolder, bins = 18,dist='normal')[3:5]
    print("Noise peak to peak range limits (+- 2*sigma = 95.45%%) S1 (mu | 4s ): %f | %f "%(mu_1,4*std_1))
    print("Noise peak to peak range limits (+- 2*sigma = 95.45%%) S2 (mu | 4s ): %f | %f "%(mu_2,4*std_2))