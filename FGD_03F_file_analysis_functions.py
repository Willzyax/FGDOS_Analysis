from FGD_03F_functions import *

# functions for per file analysis
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def A1_HIGH2_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS A1_HIGH2 ********************************************")
    print("interesting: sensitivity reference")
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_1216\\A1_HIGH2'
    file_name_1, file_name_2 = str(file_1), str(file_2)

    # linear range little bit extended for S1 , start at t = 1e5, end at t = 65e4 ms
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    LET = LET_70_Si
    Flux = A1[1]
    dose_rate = LET*Flux*gy_conv # Gy/s
    i_start_2, i_end_2 = dataset_2.index[dataset_2[t_info] >= 1e5].tolist()[0], dataset_2.index[dataset_2[t_info] >= 65e4].tolist()[0]
    i_start_1, i_end_1 = dataset_1.index[dataset_1[t_info] >= 1e5].tolist()[0], dataset_1.index[dataset_1[t_info] >= 65e4].tolist()[0]

    # plot temperature to see of compensation might be needed. Also plot reference to see change over there (1 reading is of so ymax is changed)
    steps = 15
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    xmin, xmax= math.floor(min([min(x_1),min(x_1)])),math.ceil(max([max(x_1),max(x_1)]))
    x = [x_1,x_2]
    y = [y_1,y_2]
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[xmin,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[dataset_1[F_r_info]<100000][F_r_info],dataset_2[F_r_info]
    names = ['F reference sensor 1','F reference sensor 2']
    plot_2subs_datasets(x_1,x_2,y_1,y_2,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder)

    # determine sensitivity via linreg over (slightly extended) linear range
    index_ranges =  [[i_start_1, i_end_1,i_start_2, i_end_2]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    print("linear regression coefficients (S1 | S2) (Hz/s): %f | %f"%(coeffs[0],coeffs[1]))
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over linear range (linreg) (each sensor is a row) (S1 || S2) (Hz/Gy): ")
    print(sensitivities)
    print("p-values of linreg fits for intercept and coeff (S1 || S2): ")
    print(pvalues)

    # plot data and linreg over it
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[F_s_info],dataset_2[F_s_info]
    coeff_1, intercept_1 = coeffs[0], intercepts[0]
    coeff_2, intercept_2 = coeffs[1], intercepts[1]
    fig_1, ax_1 = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_1, y=y_1,color='indianred',label = 'Sensor 1',ax=ax_1,marker='o')
    sb.lineplot(x=x_1, y=f_linreg(coeff_1,intercept_1,x_1),color='darkred', label = str(int(sensitivities[0]))+' (Hz/Gy)',ax=ax_1,linestyle='-',alpha=0.6)
    sb.scatterplot(x=x_2, y=y_2,color='royalblue',label = 'Sensor 2',ax=ax_1,marker='o')
    sb.lineplot(x=x_2, y=f_linreg(coeff_2,intercept_2,x_2),color='darkblue', label = str(int(sensitivities[1]))+' (Hz/Gy)',ax=ax_1,linestyle='-',alpha=0.6)
    step = round((xmax-xmin)/steps)
    ax_1.set_xticks(range(xmin,xmax,step))
    ax_1.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax_1.grid()
    # fig_1.suptitle(file_name_1[0:-7]+'_linreg')
    fig_1.tight_layout()
    fig_1.show()
    fig_1.savefig(savefolder+'\\'+file_name_1[0:-7]+'_linreg'+'.png')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def A2_HIGH_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS A2_HIGH ********************************************")
    print("interesting: sensitivity degradation after recharge, sensitivity increase with flux")
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_1216\\A2_HIGH'
    file_name_1, file_name_2 = str(file_1), str(file_2)

    # linear range little bit extended for S1 , start at t = 7e4, end at t = 65e4 ms. S1 and S2 have a weird data point second measurement after recharge. 
    # S1 has extreme value during first run which is dropped
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1 = dataset_1.loc[(dataset_1[F_s_info]<100000)]
    LET = LET_70_Si
    Flux = A2[1]
    dose_rate = LET*Flux*gy_conv # Gy/s
    F_lin_min, F_lin_max = 50000, 90000 # Hz
    i_r_start_2, i_r_end_2 = dataset_2.index[dataset_2[R_r_info] == 128].tolist()[0], dataset_2.index[dataset_2[R_r_info] ==1].tolist()[0]
    i_start_2, i_end_2 = dataset_2.index[dataset_2[t_info] >= 7e4].tolist()[0], dataset_2.index[dataset_2[t_info] >= 65e4].tolist()[0]
    i_r_start_1, i_r_end_1 = dataset_1.index[dataset_1[R_r_info] == 128].tolist()[0], dataset_1.index[dataset_1[R_r_info] ==1].tolist()[0]
    i_start_1, i_end_1 = dataset_1.index[dataset_1[t_info] >= 7e4].tolist()[0], dataset_1.index[dataset_1[t_info] >= 65e4].tolist()[0]

    # plot temperature to see of compensation might be needed. Also plot reference to see change over there (1 reading is of so ymax is changed)
    steps = 15
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    xmin, xmax= math.floor(min([min(x_1),min(x_2)])),math.ceil(max([max(x_1),max(x_2)]))
    x = [x_1,x_2]
    y = [y_1,y_2]
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[xmin,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[dataset_1[F_r_info]<100000][F_r_info],dataset_2[F_r_info]
    names = ['F reference sensor 1','F reference sensor 2']
    plot_2subs_datasets(x_1,x_2,y_1,y_2,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder)

    # # F sensor overall plot
    # x = [dataset_1[t_info],dataset_2[t_info]]
    # y = [dataset_1[F_s_info],dataset_2[F_s_info]]
    # names = ['F sensor 1','F sensor 2']
    # fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,xlim=[0,max(x[1])],ylim=[40000,90000],colors=['red','blue'],
    #         markers=['o','o'])

    #  recharge properties
    t_recharge_1, t_recharge_2 = (dataset_1[t_info][i_r_start_1]-dataset_1[t_info][i_r_end_1])/1000 , (dataset_2[t_info][i_r_start_2]-dataset_2[t_info][i_r_end_2])/1000 
    print("time for a recharge (S1 | S2) (s): %f | %f "%(t_recharge_1, t_recharge_2))
    print("charge lost during recharge (S1 | S2) (Gy): %f | %f"%(t_recharge_1*dose_rate, t_recharge_2*dose_rate))

    # determine sensitivity via linreg over linear range and prepare to plot
    fig_1, ax_1 = plt.subplots(figsize=(15,7))
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[F_s_info],dataset_2[F_s_info]
    index_ranges =  [[i_start_1, i_r_start_1,i_start_2, i_r_start_2]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over linear range (linreg) (each sensor is a row) (S1 || S2) (Hz/Gy):")
    print(sensitivities)
    print("p-values of linreg fits for intercept and coeff (S1 || S2): %.30f | %f || %f | %f"%(pvalues[0][0],pvalues[0][1],pvalues[1][0],pvalues[1][1]))
    print("R squared values of linreg (S1 || S2): %f || %f"%(rsquared[0],rsquared[1]))
    coeff_1, intercept_1 = coeffs[0], intercepts[0]
    coeff_2, intercept_2 = coeffs[1], intercepts[1]
    sb.scatterplot(x=x_1, y=y_1,color='salmon',label = 'Sensor 1',ax=ax_1,marker='o',alpha=0.5)
    sb.lineplot(x=x_1, y=f_linreg(coeff_1,intercept_1,x_1),color='darkred', label = str(int(sensitivities[0]))+' (Hz/Gy)',ax=ax_1,linestyle='-')
    sb.scatterplot(x=x_2, y=y_2,color='mediumslateblue',label = 'Sensor 2',ax=ax_1,marker='o',alpha=0.5)
    sb.lineplot(x=x_2, y=f_linreg(coeff_2,intercept_2,x_2),color='darkblue', label = str(int(sensitivities[1]))+' (Hz/Gy)',ax=ax_1,linestyle='-')

    # determine sens after recharge via linreg but first discard whips by setting max dF and discard recharge overshoot
    dF_1 = max_change(dataset_1,i_start_1,i_r_start_1)
    dataset_1_new = drop_shootouts(dataset_1,i_r_end_1+20,i_end_1,dF_1)
    dF_2 = max_change(dataset_2,i_start_2,i_r_start_2)
    dataset_2_new = drop_shootouts(dataset_2,i_r_end_2+20,i_end_2,dF_2)
    # x,y = [dataset_1_new[t_info],dataset_2_new[t_info]] , [dataset_1_new[F_s_info],dataset_2_new[F_s_info]]
    # plot_xy(x,y,file_name_1[0:-7]+'_Fs_modified',['Sensor 1','Sensor 2'],steps,savefolder=savefolder)
    index_ranges =  [[dataset_1_new.index.tolist()[0],dataset_1_new.index.tolist()[-1],dataset_2_new.index.tolist()[0],dataset_2_new.index.tolist()[-1]]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1_new,dataset_2_new])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over SECOND linear range (linreg) (each sensor is a row) (S1 || S2) (Hz/Gy): ")
    print(sensitivities)
    coeff_1, intercept_1 = coeffs[0], intercepts[0]
    coeff_2, intercept_2 = coeffs[1], intercepts[1]
    sb.lineplot(x=x_1, y=f_linreg(coeff_1,intercept_1,x_1),color='indianred', label = str(int(sensitivities[0]))+' (Hz/Gy)',ax=ax_1,linestyle='-')
    sb.lineplot(x=x_2, y=f_linreg(coeff_2,intercept_2,x_2),color='royalblue', label = str(int(sensitivities[1]))+' (Hz/Gy)',ax=ax_1,linestyle='-')
    
    # show plot overall data and linregs
    ymin, ymax = limits([y_1,y_2])
    ax_1.set_ylim([ymin,ymax])
    ax_1.set_xlim([xmin,xmax])
    step = round((xmax-xmin)/steps)
    ax_1.set_xticks(range(xmin,xmax,step))
    ax_1.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax_1.grid()
    # fig_1.suptitle(file_name_1[0:-7]+'_linreg')
    fig_1.tight_layout()
    fig_1.show()
    fig_1.savefig(savefolder+'\\'+file_name_1[0:-7]+'_linreg'+'.png')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def A2_LOW_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS A2_HIGH ********************************************")
    print("interesting: sensitivity reference for LOW")
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_1216\\A2_LOW'
    file_name_1, file_name_2 = str(file_1), str(file_2)

    # linear range little bit extended for S1 , start at t = 25e3, end at t = 6e5 ms
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    LET = LET_70_Si
    Flux = A2[1]
    dose_rate = LET*Flux*gy_conv # Gy/s
    F_lin_min, F_lin_max = 140000, 180000 # Hz
    i_start_1, i_end_1 = dataset_1.index[dataset_1[t_info] >= 25e3].tolist()[0], dataset_1.index[dataset_1[t_info] >= 6e5].tolist()[0]
    i_start_2, i_end_2 = dataset_2.index[dataset_2[t_info] >= 25e3].tolist()[0], dataset_2.index[dataset_2[t_info] >= 6e5].tolist()[0]

    # plot temperature to see of compensation might be needed. Also plot reference to see change over there (1 reading is of so ymax is changed)
    steps = 15
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    xmin, xmax= math.floor(min([min(x_1),min(x_2)])),math.ceil(max([max(x_1),max(x_2)]))
    x = [x_1,x_2]
    y = [y_1,y_2]
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[xmin,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[F_r_info],dataset_2[F_r_info]
    names = ['F reference sensor 1','F reference sensor 2']
    plot_2subs_datasets(x_1,x_2,y_1,y_2,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder)

    # # F sensor overall plot
    # x = [dataset_1[t_info],dataset_2[t_info]]
    # y = [dataset_1[F_s_info],dataset_2[F_s_info]]
    # names = ['F sensor 1','F sensor 2']
    # fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Fs',names,steps,savefolder=savefolder,xlim=[0,max(x[1])],ylim=[40000,90000],colors=['red','blue'],
    #         markers=['o','o'])

    # determine sensitivity via linreg over linear range and prepare to plot
    fig_1, ax_1 = plt.subplots(figsize=(15,7))
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[F_s_info],dataset_2[F_s_info]
    index_ranges =  [[i_start_1, i_end_1,i_start_2, i_end_2]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over linear range (linreg) (each sensor is a row) (S1 || S2) (Hz/Gy): ")
    print(sensitivities)
    print("p-values of linreg fits for intercept and coeff (S1 || S2): ")
    print(pvalues)
    coeff_1, intercept_1 = coeffs[0], intercepts[0]
    coeff_2, intercept_2 = coeffs[1], intercepts[1]
    sb.scatterplot(x=x_1, y=y_1,color='salmon',label = 'Sensor 1',ax=ax_1,marker='o',alpha=0.5)
    sb.lineplot(x=x_1, y=f_linreg(coeff_1,intercept_1,x_1),color='darkred', label = str(int(sensitivities[0]))+' (Hz/Gy)',ax=ax_1,linestyle='-')
    sb.scatterplot(x=x_2, y=y_2,color='mediumslateblue',label = 'Sensor 2',ax=ax_1,marker='o',alpha=0.5)
    sb.lineplot(x=x_2, y=f_linreg(coeff_2,intercept_2,x_2),color='darkblue', label = str(int(sensitivities[1]))+' (Hz/Gy)',ax=ax_1,linestyle='-')
    
    # show plot overall data and linregs
    ymin, ymax = math.floor(min([min(y_1),min(y_2)])),math.ceil(max([max(y_1),max(y_2)]))
    ax_1.set_ylim([ymin,ymax])
    ax_1.set_xlim([xmin,xmax])
    step = round((xmax-xmin)/steps)
    ax_1.set_xticks(range(xmin,xmax,step))
    ax_1.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax_1.grid()
    # fig_1.suptitle(file_name_1[0:-7]+'_linreg')
    fig_1.tight_layout()
    fig_1.show()
    fig_1.savefig(savefolder+'\\'+file_name_1[0:-7]+'_linreg'+'.png')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def A3_HIGH_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS A3_HIGH ********************************************")
    print("interesting: data whips, overshoots after recharge, low sensitivity")
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_1216\\A3_HIGH'
    file_name_1, file_name_2 = str(file_1), str(file_2)

    # determine start and end of recharges via recharge register :) (note there is 1 weird value for sensor 1, first frop this)
    # there are overshoots after recharge, discard these as well by skipping 5 values for sensitivity calculation
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dataset_1.drop(dataset_1.index[dataset_1[F_s_info]>100000].tolist()[0],inplace=True,axis=0)
    LET = LET_70_Si
    Flux = A3[1]
    dose_rate = LET*Flux*gy_conv # Gy/s
    F_lin_min, F_lin_max = 50000, 90000 # Hz
    i_r_start_2_1, i_r_end_2_1 = dataset_2.index[dataset_2[R_r_info] == 128].tolist()[0], dataset_2.index[dataset_2[R_r_info] == 1].tolist()[5]
    i_r_start_2_2, i_r_end_2_2 = dataset_2.index[dataset_2[R_r_info] == 129].tolist()[0], dataset_2.index[dataset_2[R_r_info] == 2].tolist()[5]
    i_r_start_1, i_r_end_1 = dataset_1.index[dataset_1[R_r_info] == 128].tolist()[0], dataset_1.index[dataset_1[R_r_info] == 1].tolist()[5]
    t_start_2_1, t_end_2_1 = dataset_2[t_info][i_r_start_2_1], dataset_2[t_info][i_r_end_2_1]
    t_start_2_2, t_end_2_2 = dataset_2[t_info][i_r_start_2_2], dataset_2[t_info][i_r_end_2_2] 
    t_start_1, t_end_1, t_min_1 = dataset_1[t_info][i_r_start_1], dataset_1[t_info][i_r_end_1], dataset_1[t_info].iloc[-1]

    # plot temperature to see of compensation might be needed. Also plot reference to see change over there (1 reading is of so ymax is changed)
    steps = 15
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    xmin, xmax= math.floor(min([min(x_1),min(x_1)])),math.ceil(max([max(x_1),max(x_1)]))
    x = [x_1,x_2]
    y = [y_1,y_2]
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[xmin,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[dataset_1[F_r_info]<100000][F_r_info],dataset_2[F_r_info]
    names = ['F reference sensor 1','F reference sensor 2']
    plot_2subs_datasets(x_1,x_2,y_1,y_2,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder)

    # determine sensitivity via endpoints. Use recharge times to determine recharge duration for each sensor
    dose_1 = dose_rate*(t_min_1-t_end_1)/1000 # Gy
    dose_2 = dose_rate*(t_start_2_2-t_end_2_1)/1000
    dF_1 = dataset_1[F_s_info][i_r_end_1] - dataset_1[F_s_info].iloc[-1]
    dF_2 = dataset_2[F_s_info][i_r_end_2_1] - dataset_2[F_s_info][i_r_start_2_2]
    sens_A3_HIGH_1, sens_A3_HIGH_2 = dF_1/dose_1 , dF_2/dose_2 # Hz/Gy
    t_recharge_1_1, t_recharge_2_1, t_recharge_2_2 = (t_end_1-t_start_1)/1000,(t_end_2_1-t_start_2_1)/1000,(t_end_2_2-t_start_2_2)/1000 # s
    print("sensitivity over whole range (endpoints) (S1 | S2) (Hz/Gy): %f | %f"%(sens_A3_HIGH_1,sens_A3_HIGH_2))
    print("time for a recharge (S1 | S2 | S2) (s): %f | %f | %f"%(t_recharge_1_1, t_recharge_2_1, t_recharge_2_2))
    print("charge lost during recharge (S1 | S2 | S2) (Gy): %f | %f | %f"%(t_recharge_1_1*dose_rate, t_recharge_2_1*dose_rate, t_recharge_2_2*dose_rate))
    
    # determine sensitivity via linear regression
    # x_1, x_2 = sm.add_constant( dataset_1[t_info][i_r_end_1:-1]), sm.add_constant( dataset_2[t_info][i_r_end_2_1:i_r_start_2_2])
    # y_1, y_2 = dataset_1[F_s_info][i_r_end_1:-1], dataset_2[F_s_info][i_r_end_2_1:i_r_start_2_2]
    # model_1, model_2 = sm.OLS(y_1,x_1).fit(), sm.OLS(y_2,x_2).fit()
    # print(model_1.summary())
    # intercept_1, coeff_1 = model_1.params
    # intercept_2, coeff_2 = model_2.params
    # p_intercept_1, p_coeff_1 = model_1.pvalues 
    # p_intercept_2, p_coeff_2 = model_2.pvalues
    # sens_A3_HIGH_1, sens_A3_HIGH_2 = -coeff_1/dose_rate*1000 , -coeff_2/dose_rate*1000 # Hz/Gy
    # print("sensitivity over whole range (linear regression) (S1 | S2) (Hz/Gy): %f | %f"%(sens_A3_HIGH_1,sens_A3_HIGH_2))
    # print("dF_1 according to endpoints: %f | linreg: %f"%(dF_1,(coeff_1*t_end_1-coeff_1*t_min_1)))

    # determine sensitivity via linreg over linear range only, first eliminate whips. dF_1 was chosen for dataset 2 as well
    # because of the limited amount of data for 2 (also: dF_2 was lower and should be higher because of higher sens)
    # linreg plots as well over data without whips
    dF_1 = max_change(dataset_1,0,i_r_start_1)
    dataset_1_new = drop_shootouts(dataset_1,i_r_end_1+20,-2,dF_1)
    dF_2 = max_change(dataset_2,0,i_r_start_2_1)
    if (dF_2<dF_1): dF_2 = dF_1
    dataset_2_new = drop_shootouts(dataset_2,i_r_end_2_1+20,i_r_start_2_2,dF_2)
    i_lr_start_1, i_lr_end_1, i_lr_start_2, i_lr_end_2 = dataset_1_new.index[(dataset_1_new[F_s_info] <= F_lin_max)].tolist()[0], (
                dataset_1_new.index[(dataset_1_new[F_s_info] <= F_lin_min)].tolist()[0]), (
                dataset_2_new.index[(dataset_2_new[F_s_info] <= F_lin_max)].tolist()[0]), (
                dataset_2_new.index[(dataset_2_new[F_s_info] <= F_lin_min)].tolist()[0])
    index_ranges =  [[i_lr_start_1,i_lr_end_1,i_lr_start_2,i_lr_end_2]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over linear range (linreg without whips) (each sensor is a row) (S1 || S2) (Hz/Gy): ")
    print(sensitivities)
    print("p-values of linreg fits for intercept and coeff (S1 || S2): ")
    print(pvalues)
    x_1,x_2,y_1,y_2 = dataset_1_new[t_info],dataset_2_new[t_info],dataset_1_new[F_s_info],dataset_2_new[F_s_info]
    fig_1, ax_1 = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_1, y=y_1,color='salmon',label = 'Sensor 1',ax=ax_1,marker='o',alpha=0.5)
    sb.lineplot(x=x_1, y=f_linreg(coeffs[0],intercepts[0],x_1),color='darkred', label = str(int(sensitivities[0]))+' (Hz/Gy)',ax=ax_1,linestyle='-')
    sb.scatterplot(x=x_2, y=y_2,color='mediumslateblue',label = 'Sensor 2',ax=ax_1,marker='o',alpha=0.5)
    sb.lineplot(x=x_2, y=f_linreg(coeffs[1],intercepts[1],x_2),color='darkblue', label = str(int(sensitivities[1]))+' (Hz/Gy)',ax=ax_1,linestyle='-')
    step = round((xmax-xmin)/steps)
    ax_1.set_xticks(range(xmin,xmax,step))
    ax_1.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax_1.grid()
    # fig_1.suptitle(file_name_1[0:-7]+'_linreg_linrange')
    fig_1.tight_layout()
    fig_1.show()
    fig_1.savefig(savefolder+'\\'+file_name_1[0:-7]+'_linreg_linrange'+'.png')

    # plot data, F sens and F ref
    steps = 15
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[F_s_info],dataset_2[F_s_info]
    # for overshoot : 
    steps = 80
    fig,ax = plot_xy([x_2],[y_2],file_name_1,['Sensor 2'],steps,savefolder=savefolder,xlim=[20200,40200],ylim=[106000,111000],colors=['blue'])
    plt.axvline(x=dataset_2.loc[dataset_2[R_r_info] == 1,t_info].iloc[0])
    fig.savefig(savefolder+'\\'+file_name_1[:-7]+'_overshoot.png')
    steps = 15
    # for end of discharge:
    # plot_datasets(x_1,x_2,y_1,y_2,file_name_1[0:-7]+'_discharge',steps,savefolder=savefolder,xlim=[530000,550000],ylim=[20000,30000])
    # for weird data wips : 
    # fig,ax = plot_datasets(x_1,x_2,y_1,y_2,file_name_1[0:-7]+'_whips',steps,savefolder=savefolder,xlim=[20200,120000],ylim=[90000,111000])
    # for overall view with added reference frequency
    x = [dataset_1[t_info],dataset_1[t_info],dataset_2[t_info],dataset_2[t_info]]
    y = [dataset_1[F_s_info],dataset_1[F_r_info],dataset_2[F_s_info],dataset_2[F_r_info]]
    names = ['F sensor 1','F reference 1','F sensor 2','F reference 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Fs&r',names,steps,savefolder=savefolder,xlim=[0,max(x[1])],ylim=[20000,115000],colors=['darkred','indianred','darkblue','royalblue'],
            markers=['o','o','o','o'])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def A3_LOW_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS A3_LOW ********************************************")
    print('interesting: low sensitivity, degradation during RF trip, change in sensitivity after RF trip')
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_1216\\A3_LOW'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    LET = LET_70_Si
    Flux = A3[1]
    dose_rate = LET*Flux*gy_conv # Gy/s
    F_lin_min, F_lin_max = 140000, 180000
    # approx: beam starts at t=7e4 (i=), stops at t=4e5 (i=), starts at t=605e3 (i=), stops at t=111e4 (i=)
    i_break_start_2, i_break_end_2 = dataset_2.index[dataset_2[t_info] >= 4e5].tolist()[0], dataset_2.index[dataset_2[t_info] >= 605e3].tolist()[0]
    i_beam_start_2, i_beam_end_2 = dataset_2.index[dataset_2[t_info] >= 7e4].tolist()[0], dataset_2.index[dataset_2[t_info] >= 111e4].tolist()[0]
    i_recharge_start_1, i_recharge_end_1 = dataset_1.index[dataset_1[R_r_info]>0].tolist()[0], dataset_1.index[dataset_1[R_r_info]==1].tolist()[0]
    i_break_start_1, i_break_end_1 = dataset_1.index[dataset_1[t_info] >= 4e5].tolist()[0], dataset_1.index[dataset_1[t_info] >= 605e3].tolist()[0]
    i_beam_start_1, i_beam_end_1 = dataset_1.index[dataset_1[t_info] >= 7e4].tolist()[0], dataset_1.index[dataset_1[t_info] >= 111e4].tolist()[0]
    # di_rftrip_1, di_rftrip_2 = i_break_end_1-i_break_start_1, i_break_end_2-i_break_start_2
    t_start_1_1, t_end_1_1 = dataset_1[t_info][i_beam_start_1], dataset_1[t_info][i_break_start_1]
    t_start_1_2, t_end_1_2 = dataset_1[t_info][i_break_end_1], dataset_1[t_info][i_beam_end_1]
    t_start_2_1, t_end_2_1 = dataset_2[t_info][i_beam_start_2], dataset_2[t_info][i_break_start_2]
    t_start_2_2, t_end_2_2 = dataset_2[t_info][i_break_end_2], dataset_2[t_info][i_beam_end_2]

    # plot temperature to see of compensation might be needed (together with reference frequency to see influence)
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    xmin, xmax = math.floor(min([x_1.min(),x_2.min()])), math.ceil(max([x_1.max(),x_2.max()]))
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = math.floor(min([min(y_1_r),min(y_2_r)])), math.ceil(max([max(y_1_r),max(y_2_r)]))
    names = ['F reference 1','F reference 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[ymin,ymax],colors=['darkred','darkblue'])

    # plot overall view Fr and Fs
    steps = 30
    x = [x_1,x_1,x_2,x_2]
    y_1_s, y_2_s = dataset_1[F_s_info], dataset_2[F_s_info]
    y = [y_1_s,y_1_r,y_2_s,y_2_r]
    # xmin, xmax = [0,800000]
    ymin, ymax = [130000,190000] # math.floor(min([min(y_1_s),min(y_2_s)])), math.ceil(max([max(y_1_s),max(y_2_s)]))
    names = ['F sensor 1','F reference 1','F sensor 2','F reference 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Fs&r',names,steps,savefolder=savefolder,colors=['darkred','indianred','darkblue','royalblue'],
                    markers=['o','.','o','.'])

    # Use recharge of Sensor 1 to determine t_recharge. Determine RF trip times
    t_recharge_start, t_recharge_end = dataset_1[t_info][i_recharge_start_1], dataset_1[t_info][i_recharge_end_1]
    t_recharge_2 = (t_recharge_end-t_recharge_start)/1000
    print("time for a recharge (S1) (s): %f"%(t_recharge_2))
    print("charge lost during recharge (S2) (Gy): %f"%(t_recharge_2*dose_rate))
    dt_rftrip_1, dt_rftrip_2 = t_start_1_2-t_end_1_1, t_start_2_2-t_end_2_1
    print("cyclotron RF trip times (S1 , S2): %f | %f"%(dt_rftrip_1,dt_rftrip_2))

    # determine sensitivity via linreg over linear range only (dataset is modified to exlude RF trip, and split because of difference in linear response)
    index_ranges = [[i_beam_start_1,i_break_start_1,i_beam_start_2,i_break_start_2],[i_break_end_1,i_recharge_start_1,i_break_end_2,i_beam_end_2]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over linear range (linreg) (S1a | S1b || S2a | S2b) (Hz/Gy): ")
    print(sensitivities)
    print("p-values of linreg fits for intercept and coeff (S1a | S1b | S2a | S2b) (Hz/Gy): ")
    print(pvalues)

    # plot linreg sensitivities
    # xmin, xmax = math.floor(dataset_1[t_info][i_beam_start_1]), math.ceil(dataset_1[t_info][i_beam_end_1])
    # ymin, ymax = math.floor(dataset_1[F_s_info][i_beam_start_1]), math.ceil(dataset_1[F_s_info][i_beam_end_1])
    step = round((xmax-xmin)/steps)
    x_linreg = range(xmin,xmax,500)

    fig, ax = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_1, y=y_1_s,color='salmon',label = 'Sensor 1',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[0][0],intercepts[0][0],x_linreg),color='indianred', label = str(int(sensitivities[0][0]))+' (Hz/Gy)',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[0][1],intercepts[0][1],x_linreg),color='darkred', label = str(int(sensitivities[0][1]))+' (Hz/Gy)',ax=ax)
    ax.set_xticks(range(xmin,xmax,step))
    ax.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax.grid()
    fig.suptitle(file_name_1[0:-7]+'_s1_linreg')
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_1[0:-7]+'_s1_linreg'+'.png')

    fig, ax = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_2, y=y_2_s,color='mediumslateblue',label = 'Sensor 2',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][0],intercepts[1][0],x_linreg),color='royalblue', label = str(int(sensitivities[1][0]))+' (Hz/Gy)',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][1],intercepts[1][1],x_linreg),color='darkblue', label = str(int(sensitivities[1][1]))+' (Hz/Gy)',ax=ax)
    ax.set_xticks(range(xmin,xmax,step))
    ax.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax.grid()
    fig.suptitle(file_name_1[0:-7]+'_s2_linreg')
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_1[0:-7]+'_s2_linreg'+'.png')

    # us linreg to determine F change during brake. Set an offset because otherwise some data during testing is taken into account
    offset_start, offset_end = 20, 20
    x_1, x_2 = dataset_1[t_info][i_break_start_1+offset_start:i_break_end_1-offset_end], dataset_2[t_info][i_break_start_2+offset_start:i_break_end_2-offset_end]
    y_1,y_2 = dataset_1[F_s_info][i_break_start_1+offset_start:i_break_end_1-offset_end], dataset_2[F_s_info][i_break_start_2+offset_start:i_break_end_2-offset_end]
    coeff_1, intercept_1 = np.polyfit(x_1.to_numpy(),y_1.to_numpy(),1)
    coeff_2, intercept_2 = np.polyfit(x_2.to_numpy(),y_2.to_numpy(),1)
    print('coeff of linreg during break (S1 | S2) (Hz/s): %f | %f'%(coeff_1,coeff_2))

    #  plot off break and linreg
    fig, axs = plt.subplots(nrows=2,figsize=(15,7))
    sb.scatterplot(x=x_1, y=y_1,color='salmon',label = 'Sensor 1',ax=axs[0])
    sb.lineplot(x=x_1, y=f_linreg(coeff_1,intercept_1,x_1),color='darkred', label = format(coeff_1*60*60*1000,'.0f')+' (Hz/hr)',ax=axs[0])
    sb.scatterplot(x=x_2, y=y_2,color='mediumslateblue',label = 'Sensor 2',ax=axs[1])
    sb.lineplot(x=x_2, y=f_linreg(coeff_2,intercept_2,x_2),color='darkblue', label =  format(coeff_2*60*60*1000,'.0f')+' (Hz/hr)',ax=axs[1])
    xmin, xmax= math.floor(min([min(x_1),min(x_1)])),math.ceil(max([max(x_1),max(x_1)]))
    step = round((xmax-xmin)/steps)
    axs[0].set_xticks(range(xmin,xmax,step)), axs[1].set_xticks(range(xmin,xmax,step))
    axs[0].set_xticklabels(range(xmin,xmax,step),rotation=30), axs[1].set_xticklabels(range(xmin,xmax,step),rotation=30)
    axs[0].grid(), axs[1].grid()
    fig.suptitle(file_name_1[0:-7]+'_degradation')
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_1[0:-7]+'_degradation'+'.png')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def A4_HIGH_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS A4_HIGH ********************************************")
    print("interesting: change in sensitivity over different parts of test | lower end of linear range")
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_1216\\A4_HIGH'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    LET = LET_70_Si
    Flux = A4[1]
    dose_rate = LET*Flux*gy_conv # Gy/s
    print(dose_rate)
    dose_rate = 0.014318
    print(dose_rate)
    # use end and startpoints. use recharge of Sensor 2 to determine t_recharge
    # approx: beam starts at t=4000 ms, stops at t=455253.8148 ms (i=), starts again at t=655556.282 ms (i=)
    i_start_1_1, i_end_1_1, i_start_1_2, i_end_1_2, i_start_1_3, i_end_1_3 = dataset_1.index[dataset_1[t_info] >= 4000].tolist()[0] , dataset_1.index[dataset_1[R_r_info] >=128
                        ].tolist()[0] , dataset_1.index[dataset_1[R_r_info] == 1].tolist()[0] , dataset_1.index[dataset_1[t_info] >= 455253].tolist()[0] , (
                        dataset_1.index[dataset_1[t_info] >= 655556].tolist()[0]) , (
                        dataset_1.loc[dataset_1[t_info] >= 655556][F_s_info].idxmin())
    i_start_2_1, i_end_2_1 , i_start_2_2, i_end_2_2, i_start_2_3, i_end_2_3 = dataset_2.index[dataset_2[t_info] >= 4000].tolist()[0] , dataset_2.index[dataset_2[R_r_info] >=128
                        ].tolist()[0] , dataset_2.index[dataset_2[R_r_info] == 1].tolist()[0] , dataset_2.index[dataset_2[t_info] >= 455253].tolist()[0], (
                        dataset_2.index[dataset_2[t_info] >= 655556].tolist()[0]) , (
                        dataset_2.loc[dataset_2[t_info] >= 655556][F_s_info].idxmin())
    t_start_1_1, t_end_1_1, t_start_1_2, t_end_1_2, t_start_1_3, t_end_1_3 = dataset_1[t_info][i_start_1_1] , dataset_1[t_info][i_end_1_1], dataset_1[t_info][i_start_1_2] , (
                        dataset_1[t_info][i_end_1_2]), dataset_1[t_info][i_start_1_3], dataset_1[t_info][i_end_1_3]
    t_start_2_1, t_end_2_1, t_start_2_2, t_end_2_2, t_start_2_3, t_end_2_3 = dataset_2[t_info][i_start_2_1] , dataset_2[t_info][i_end_2_1], dataset_2[t_info][i_start_2_2] , (
                        dataset_2[t_info][i_end_2_2]), dataset_2[t_info][i_start_2_3], dataset_2[t_info][i_end_2_3]
    # also determine linear range endpoints (only recharge register is be used but 1 weird data point messes this up for sensor 1... Hence use 2nd value)
    i_start_1_1_linear, i_end_1_1_linear, i_start_1_2_linear, i_end_1_2_linear = dataset_1.index[dataset_1[F_s_info] <= 90000].tolist()[0], (
                        dataset_1.index[dataset_1[F_s_info] <= 50000].tolist()[0]), dataset_1.index[(dataset_1[F_s_info] <= 90000) & (dataset_1[R_r_info] == 1)].tolist()[1], (
                        dataset_1.index[(dataset_1[F_s_info] <= 50000) & (dataset_1[R_r_info] == 1)].tolist()[0])
    i_start_2_1_linear, i_end_2_1_linear, i_start_2_2_linear, i_end_2_2_linear = dataset_2.index[dataset_2[F_s_info] <= 90000].tolist()[0], (
                        dataset_2.index[dataset_2[F_s_info] <= 50000].tolist()[0]), dataset_2.index[(dataset_2[F_s_info] <= 90000) & (dataset_2[R_r_info] == 1)].tolist()[0], (
                        dataset_2.index[(dataset_2[F_s_info] <= 50000) & (dataset_2[R_r_info] == 1)].tolist()[0])
    t_start_1_1_linear, t_end_1_1_linear, t_start_1_2_linear, t_end_1_2_linear = dataset_1[t_info][i_start_1_1_linear], dataset_1[t_info][i_end_1_1_linear],(
                        dataset_1[t_info][i_start_1_2_linear]), dataset_1[t_info][i_end_1_2_linear]
    t_start_2_1_linear, t_end_2_1_linear, t_start_2_2_linear, t_end_2_2_linear = dataset_2[t_info][i_start_2_1_linear], dataset_2[t_info][i_end_2_1_linear],(
                        dataset_2[t_info][i_start_2_2_linear]), dataset_2[t_info][i_end_2_2_linear]


    # plot temperature to see of compensation might be needed. Also plot reference to see change over there (1 reading is of so ymax is changed)
    steps = 15
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits([x_1,x_2])
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[dataset_1[F_r_info]<500000][F_r_info],dataset_2[F_r_info]
    names = ['F reference sensor 1', 'F reference sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'])

    # linear range sensitivities using endpoints, recharge times and lost dose
    dose_1_1, dose_1_2 = dose_rate*(t_end_1_1_linear-t_start_1_1_linear)/1000, dose_rate*(t_end_1_2_linear-t_start_1_2_linear-(t_start_1_3-t_end_1_2))/1000
    dose_2_1, dose_2_2 = dose_rate*(t_end_2_1_linear-t_start_2_1_linear)/1000, dose_rate*(t_end_2_2_linear-t_start_2_2_linear-(t_start_2_3-t_end_2_2))/1000
    dF_1_1, dF_1_2 = dataset_1[F_s_info][i_start_1_1_linear] - dataset_1[F_s_info][i_end_1_1_linear] , (
                        dataset_1[F_s_info][i_start_1_2_linear] - dataset_1[F_s_info][i_end_1_2_linear])
    dF_2_1, dF_2_2 = dataset_2[F_s_info][i_start_2_1_linear] - dataset_2[F_s_info][i_end_2_1_linear] , (
                        dataset_2[F_s_info][i_start_2_2_linear] - dataset_2[F_s_info][i_end_2_2_linear])
    sens_A4_LOW_1_1, sens_A4_LOW_1_2 = dF_1_1/dose_1_1 , (dF_1_2)/dose_1_2
    sens_A4_LOW_2_1, sens_A4_LOW_2_2 = dF_2_1/dose_2_1 , dF_2_2/dose_2_2
    print("sensitivity over two linear ranges (endpoints) (S1) (Hz/Gy): %f | %f"%(sens_A4_LOW_1_1,sens_A4_LOW_1_2))
    print("sensitivity over two linear ranges (endpoints) (S2) (Hz/Gy): %f | %f"%(sens_A4_LOW_2_1,sens_A4_LOW_2_2))
    t_recharge_1 = -(dataset_1[t_info][i_end_1_1]-dataset_1[t_info][i_start_1_2])/1000
    t_recharge_2 = -(dataset_2[t_info][i_end_2_1]-dataset_2[t_info][i_start_2_2])/1000
    print("time for a recharge (S1 | S2) (s): %f | %f"%(t_recharge_1,t_recharge_2))
    print("charge lost during recharge (S1 | S2) (Gy): %f | %f"%(t_recharge_1*dose_rate,t_recharge_2*dose_rate))
    dt_rftrip_1, dt_rftrip_2 = (t_start_1_3-t_end_1_2)/1000, (t_start_2_3-t_end_2_2)/1000
    print("cyclotron RF trip times (S1 , S2) (s): %f | %f"%(dt_rftrip_1,dt_rftrip_2))

    # linreg during RF trip
    offset_start, offset_end = 20, 20
    x_1, x_2 = dataset_1[t_info][i_end_1_2+offset_start:i_start_1_3-offset_end], dataset_2[t_info][i_end_2_2+offset_start:i_start_2_3-offset_end]
    y_1,y_2 = dataset_1[F_s_info][i_end_1_2+offset_start:i_start_1_3-offset_end], dataset_2[F_s_info][i_end_2_2+offset_start:i_start_2_3-offset_end]
    coeff_1, intercept_1 = np.polyfit(x_1.to_numpy(),y_1.to_numpy(),1)
    coeff_2, intercept_2 = np.polyfit(x_2.to_numpy(),y_2.to_numpy(),1)
    print('coeff of linreg during break (S1 | S2) (Hz/hr): %f | %f'%(coeff_1*1000*60*60,coeff_2*1000*60*60)) # convert Hz/ms to Hz/hr
    #  plot off break and linreg
    fig, axs = plt.subplots(nrows=2,figsize=(15,7))
    sb.scatterplot(x=x_1, y=y_1,color='salmon',label = 'Sensor 1',ax=axs[0])
    sb.lineplot(x=x_1, y=f_linreg(coeff_1,intercept_1,x_1),color='darkred', label = format(coeff_1*60*60*1000,'.0f')+' (Hz/hr)',ax=axs[0])
    sb.scatterplot(x=x_2, y=y_2,color='mediumslateblue',label = 'Sensor 2',ax=axs[1])
    sb.lineplot(x=x_2, y=f_linreg(coeff_2,intercept_2,x_2),color='darkblue', label =  format(coeff_2*60*60*1000,'.0f')+' (Hz/hr)',ax=axs[1])
    xmin, xmax= math.floor(min([min(x_1),min(x_1)])),math.ceil(max([max(x_1),max(x_1)]))
    step = round((xmax-xmin)/steps)
    axs[0].set_xticks(range(xmin,xmax,step)), axs[1].set_xticks(range(xmin,xmax,step))
    axs[0].set_xticklabels(range(xmin,xmax,step),rotation=30), axs[1].set_xticklabels(range(xmin,xmax,step),rotation=30)
    axs[0].grid(), axs[1].grid()
    fig.tight_layout()
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_1[0:-7]+'_degradation'+'.png')

    # linear range sensitivities using linear regression (in three parts)
    offset_test = 0 # offset can be used to test, set to 0 for normal linreg
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[F_s_info],dataset_2[F_s_info]
    index_ranges =  [[i_start_1_1_linear,i_end_1_1_linear,i_start_2_1_linear,i_end_2_1_linear],[i_start_1_2_linear-offset_test,i_end_1_2+offset_test,i_start_2_2_linear-offset_test,i_end_2_2+offset_test],
                    [i_start_1_3,i_end_1_2_linear,i_start_2_3,i_end_2_2_linear]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over linear range (linreg) (each sensor is a row) (S1a | S1b | S1c || S2a | S2b | S2c) (Hz/Gy): ")
    print(sensitivities)
    print("p-values of linreg fits for intercept and coeff (S1a || S1b | S1c || S2a | S2b | S2c): ")
    print(pvalues)
    print("R-squared of linreg fits for intercept and coeff (S1a || S1b | S1c || S2a | S2b | S2c): ")
    print(rsquared)


    # plot linreg sensitivities over whole database graph
    xmin, xmax= math.floor(min(dataset_1[t_info])),math.ceil(max(dataset_1[t_info]*1.01))
    ymin, ymax= math.floor(min(dataset_1[F_s_info]*0.97)),math.ceil(max(dataset_1[F_s_info]*1.01))
    step = round((xmax-xmin)/steps)
    x_linreg = range(xmin,xmax,500)

    fig, ax = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_2, y=y_2,color='royalblue',label = 'Sensor 2',ax=ax, alpha=0.5)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][0],intercepts[1][0],x_linreg),color='salmon', label = str(int(sensitivities[1][0]))+' (Hz/Gy)',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][1],intercepts[1][1],x_linreg),color='darkred', label = str(int(sensitivities[1][1]))+' (Hz/Gy)',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][2],intercepts[1][2],x_linreg),color='red', label = str(int(sensitivities[1][2]))+' (Hz/Gy)',ax=ax)
    ax.set_xticks(range(xmin,xmax,step))
    ax.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax.grid()
    ax.set_ylim([ymin,ymax])
    # fig.suptitle(file_name_2[0:-7]+'_s2_linreg')
    fig.tight_layout()
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_2[0:-7]+'_s2_linreg'+'.png')

    fig, ax = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_1, y=y_1,color='indianred',label = 'Sensor 1',ax=ax, alpha=0.5)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[0][0],intercepts[0][0],x_linreg),color='mediumslateblue', label = str(int(sensitivities[0][0]))+' (Hz/Gy)',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[0][1],intercepts[0][1],x_linreg),color='darkblue', label = str(int(sensitivities[0][1]))+' (Hz/Gy)',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[0][2],intercepts[0][2],x_linreg),color='blue', label = str(int(sensitivities[0][2]))+' (Hz/Gy)',ax=ax)
    ax.set_xticks(range(xmin,xmax,step))
    ax.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax.grid()
    ax.set_ylim([ymin,ymax])
    # fig.suptitle(file_name_1[0:-7]+'_s1_linreg')
    fig.tight_layout()
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_1[0:-7]+'_s1_linreg'+'.png')

    # plot only the first part before the recharge
    x_1,x_2 = dataset_1.loc[i_start_1_1:i_end_1_1_linear,t_info],dataset_2.loc[i_start_2_1:i_end_2_1_linear,t_info]
    y_1, y_2 = dataset_1.loc[i_start_2_1:i_end_2_1_linear,F_s_info],dataset_2.loc[i_start_2_1:i_end_2_1_linear,F_s_info]
    xmin, xmax = limits([x_1,x_2])
    x_linreg = range(xmin,xmax,500)
    fig, ax = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_2, y=y_2,color='royalblue',label = 'Sensor 2',ax=ax, alpha=0.5,marker='o')
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][0],intercepts[1][0],x_linreg),color='darkred', label = str(int(sensitivities[1][0]))+' (Hz/Gy)',ax=ax)
    sb.scatterplot(x=x_1, y=y_1,color='indianred',label = 'Sensor 1',ax=ax, alpha=0.5,marker='o')
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[0][0],intercepts[0][0],x_linreg),color='darkblue', label = str(int(sensitivities[0][0]))+' (Hz/Gy)',ax=ax)
    # ax.set_xticks(range(xmin,xmax,step))
    # ax.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax.grid()
    ax.set_ylim([40000,ymax])
    # fig.suptitle(file_name_2[0:-7]+'_linreg_extended')
    fig.tight_layout()
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_2[0:-7]+'_linreg_extended'+'.png')

    # test to see difference in linreg sens value during part without RF trip to see if sens changes here as well.
    print("CHECK BY BREAKING FIRST LINEAR RANGE")
    F_s1_break, F_s2_break = dataset_1[F_s_info][i_end_1_2], dataset_2[F_s_info][i_end_2_2]
    i_break_1, i_break_2 = dataset_1.index[dataset_1[F_s_info]<=F_s1_break].tolist()[0], dataset_2.index[dataset_2[F_s_info]<=F_s2_break].tolist()[0]
    index_ranges = [[i_start_1_1_linear,i_break_1,i_start_2_1_linear,i_break_2],[i_break_1,i_end_1_1_linear,i_break_2,i_end_2_1_linear]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over linear range (linreg) (each sensor is a row) (S1a | S1b || S2a | S2b) (Hz/Gy): ")
    print(sensitivities)

    # check if linear range can be extended upwards
    print("CHECK LINEAR RANGE EXTENSION to upper limit")
    index_ranges = [[i_start_1_1,i_end_1_1_linear,i_start_2_1,i_end_2_1_linear]] 
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over extended range (linreg) (each sensor is a row) (S1 || S2) (Hz/Gy): ")
    print(sensitivities)
    # check by calculating sensitivities per 20 kHz (120 100 80 60 40 20)
    print("CHECK LINEAR RANGE EXTENSION with steps of 20 kHz from 120 to 20 kHz")
    index_ranges = [[dataset_1.index[dataset_1[F_s_info] <= 120000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 100000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 120000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 100000].tolist()[0]],[
                dataset_1.index[dataset_1[F_s_info] <= 100000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 80000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 100000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 80000].tolist()[0]],[
                dataset_1.index[dataset_1[F_s_info] <= 80000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 60000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 80000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 60000].tolist()[0]],[
                dataset_1.index[dataset_1[F_s_info] <= 60000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 40000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 60000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 40000].tolist()[0]],[
                dataset_1.index[dataset_1[F_s_info] <= 40000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 20000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 40000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 20000].tolist()[0]]               
                ]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over extended range (linreg) (each sensor is a row) (S1 || S2) (Hz/Gy): ")
    print(sensitivities)
    # check by calculating sensitivities per 10 kHz (120 100 80 60 40 20)
    print("CHECK LINEAR RANGE EXTENSION with steps of 10 kHz from 120 to 20 kHz")
    index_ranges = [[dataset_1.index[dataset_1[F_s_info] <= 120000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 110000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 120000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 110000].tolist()[0]],[
                dataset_1.index[dataset_1[F_s_info] <= 110000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 100000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 110000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 100000].tolist()[0]],[
                dataset_1.index[dataset_1[F_s_info] <= 100000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 90000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 100000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 90000].tolist()[0]],[
                dataset_1.index[dataset_1[F_s_info] <= 90000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 80000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 90000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 80000].tolist()[0]],[
                dataset_1.index[dataset_1[F_s_info] <= 80000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 70000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 80000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 70000].tolist()[0]],[
                dataset_1.index[dataset_1[F_s_info] <= 70000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 60000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 70000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 60000].tolist()[0]],[
                dataset_1.index[dataset_1[F_s_info] <= 60000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 50000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 60000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 50000].tolist()[0]]  ,[
                dataset_1.index[dataset_1[F_s_info] <= 50000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 40000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 50000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 40000].tolist()[0]]  ,[
                dataset_1.index[dataset_1[F_s_info] <= 40000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 30000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 40000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 30000].tolist()[0]]  ,[
                dataset_1.index[dataset_1[F_s_info] <= 30000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 20000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 30000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 20000].tolist()[0]]  ,[
                dataset_1.index[dataset_1[F_s_info] <= 20000].tolist()[0],dataset_1.index[dataset_1[F_s_info] <= 10000].tolist()[0],
                dataset_2.index[dataset_2[F_s_info] <= 20000].tolist()[0],dataset_2.index[dataset_2[F_s_info] <= 10000].tolist()[0]]                 
                ]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over extended range (linreg) (each sensor is a row) (S1 || S2) (Hz/Gy): ")
    print(sensitivities)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def A4_LOW_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS A4_LOW ********************************************")
    print('interesting: low sensitivity, degradation during RF trip, change in sensitivity after RF trip')
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_1216\\A4_LOW'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    LET = LET_70_Si
    Flux = A4[1]
    dose_rate = LET*Flux*gy_conv # Gy/s
    F_lin_min, F_lin_max = 140000, 180000
    # approx: beam starts at t=27000 (i=), stops at t=325500 (i=), starts at t=477700 (i=), stops at t=787000 (i=)
    i_break_start_2, i_break_end_2 = dataset_2.index[dataset_2[t_info] >= 325500].tolist()[0], dataset_2.index[dataset_2[t_info] >= 477700].tolist()[0]
    i_beam_start_2, i_beam_end_2 = dataset_2.index[dataset_2[t_info] >= 27000].tolist()[0], dataset_2.index[dataset_2[t_info] >= 787000].tolist()[0]
    i_recharge_start_2, i_recharge_end_2 = dataset_2.index[dataset_2[R_r_info]>0].tolist()[0], dataset_2.index[dataset_2[R_r_info] == 1].tolist()[0]
    i_break_start_1, i_break_end_1 = dataset_1.index[dataset_1[t_info] >= 325500].tolist()[0], dataset_1.index[dataset_1[t_info] >= 477700].tolist()[0]
    i_beam_start_1, i_beam_end_1 = dataset_1.index[dataset_1[t_info] >= 27000].tolist()[0], dataset_1.index[dataset_1[t_info] >= 787000].tolist()[0]
    t_start_1_1, t_end_1_1 = dataset_1[t_info][i_beam_start_1], dataset_1[t_info][i_break_start_1]
    t_start_1_2, t_end_1_2 = dataset_1[t_info][i_break_end_1], dataset_1[t_info][i_beam_end_1]
    t_start_2_1, t_end_2_1 = dataset_2[t_info][i_beam_start_2], dataset_2[t_info][i_break_start_2]
    t_start_2_2, t_end_2_2 = dataset_2[t_info][i_break_end_2], dataset_2[t_info][i_beam_end_2]

    # plot temperature to see of compensation might be needed (together with reference frequency to see influence)
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y)
    names = ['F reference 1','F reference 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[ymin,ymax],colors=['darkred','darkblue'])

    # use end and startpoints to determine sensitivity. Use recharge of Sensor 2 to determine t_recharge. Determine RF trip times
    dose_1 = dose_rate*(t_end_1_1-t_start_1_1+t_end_1_2-t_start_1_2)/1000
    dose_2 = dose_rate*(t_end_2_1-t_start_2_1+t_end_2_2-t_start_2_2)/1000
    dF_1 = np.mean(dataset_1[F_s_info][0:i_beam_start_1]) - np.min(dataset_1[F_s_info])
    dF_2 = np.mean(dataset_2[F_s_info][0:i_beam_start_2]) - np.min(dataset_2[F_s_info])
    sens_A4_LOW_1, sens_A4_LOW_2 = dF_1/dose_1 , dF_2/dose_2
    print("sensitivity over whole range (endpoints) (S1 | S2) (Hz/Gy): %f | %f"%(sens_A4_LOW_1,sens_A4_LOW_2)) 
    t_recharge_start = dataset_2[t_info][i_recharge_start_2]
    t_recharge_end = dataset_2[t_info][i_recharge_end_2]
    t_recharge_2 = (t_recharge_end-t_recharge_start)/1000
    print("time for a recharge (S2) (s): %f"%(t_recharge_2))
    print("charge lost during recharge (S2) (Gy): %f"%(t_recharge_2*dose_rate))
    dt_rftrip_1, dt_rftrip_2 = t_start_1_2-t_end_1_1, t_start_2_2-t_end_2_1
    print("cyclotron RF trip times (S1 , S2): %f | %f"%(dt_rftrip_1,dt_rftrip_2))

    # determine sensitivity via linreg over linear range only (dataset is modified to exlude RF trip, and split because of difference in linear response)
    index_ranges = [[i_beam_start_1,i_break_start_1,i_beam_start_2,i_break_start_2],[i_break_end_1,i_beam_end_1,i_break_end_2,i_recharge_start_2]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over linear range (linreg) (S1a | S1b || S2a | S2b) (Hz/Gy): ")
    print(sensitivities)
    print("p-values of linreg fits for intercept and coeff (S1a | S1b | S2a | S2b) (Hz/Gy): ")
    print(pvalues)

    # plot linreg sensitivities
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    step = round((xmax-xmin)/steps)
    x_linreg = range(xmin,xmax,500)

    fig, ax = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_1, y=y_1_s,color='salmon',label = 'Sensor 1',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[0][0],intercepts[0][0],x_linreg),color='royalblue', label = str(int(sensitivities[0][0]))+' (Hz/Gy)',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[0][1],intercepts[0][1],x_linreg),color='darkblue', label = str(int(sensitivities[0][1]))+' (Hz/Gy)',ax=ax)
    ax.set_xticks(range(xmin,xmax,step))
    ax.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax.grid()
    fig.suptitle(file_name_1[0:-7]+'_s1_linreg')
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_1[0:-7]+'_s1_linreg'+'.png')

    fig, ax = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_2, y=y_2_s,color='mediumslateblue',label = 'Sensor 2',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][0],intercepts[1][0],x_linreg),color='indianred', label = str(int(sensitivities[1][0]))+' (Hz/Gy)',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][1],intercepts[1][1],x_linreg),color='darkred', label = str(int(sensitivities[1][1]))+' (Hz/Gy)',ax=ax)
    ax.set_xticks(range(xmin,xmax,step))
    ax.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax.grid()
    fig.suptitle(file_name_1[0:-7]+'_s2_linreg')
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_1[0:-7]+'_s2_linreg'+'.png')
    
    # plot overall view
    steps = 30
    # x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[F_s_info],dataset_2[F_s_info]
    # plot_datasets(x_1,x_2,y_1,y_2,file_name_1[0:-7]+'_Fs&r',steps, savefolder=savefolder) # xlim=[310000,480000],ylim=[155000,169000]
    # to plot with reference frequencies
    x = [x_1,x_1,x_2,x_2]
    y = [y_1_s, y_1_r, y_2_s, y_2_r]
    xmin, xmax = [0,800000]
    ymin, ymax = [130000,190000]
    names = ['F sensor 1','F reference 1','F sensor 2','F reference 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Fs&r',names,steps,savefolder=savefolder,xlim=[xmin,xmax],ylim=[ymin,ymax],colors=['darkred','salmon','darkblue','mediumslateblue'],
                    markers=['o','.','o','.'])

    # us linreg to determine F change during brake
    x_1, x_2 = dataset_1[t_info][i_break_start_1:i_break_end_1], dataset_2[t_info][i_break_start_2:i_break_end_2]
    y_1,y_2 = dataset_1[F_s_info][i_break_start_1:i_break_end_1], dataset_2[F_s_info][i_break_start_2:i_break_end_2]
    coeff_1, intercept_1 = np.polyfit(x_1.to_numpy(),y_1.to_numpy(),1)
    coeff_2, intercept_2 = np.polyfit(x_2.to_numpy(),y_2.to_numpy(),1)
    print('coeff of linreg during break (S1 | S2) (Hz/hr): %f | %f'%(coeff_1*1000*60*60,coeff_2*1000*60*60))
    #  plot of break and linreg
    fig, axs = plt.subplots(nrows=2,figsize=(15,7))
    sb.scatterplot(x=x_1, y=y_1,color='red',label = 'Sensor 1',ax=axs[0])
    sb.lineplot(x=x_1, y=f_linreg(coeff_1,intercept_1,x_1),color='blue', label = format(coeff_1*1000*60*60,'.0f')+' (Hz/Gy)',ax=axs[0])
    sb.scatterplot(x=x_2, y=y_2,color='red',label = 'Sensor 2',ax=axs[1])
    sb.lineplot(x=x_2, y=f_linreg(coeff_2,intercept_2,x_2),color='blue', label = format(coeff_2*1000*60*60,'.0f')+' (Hz/Gy)',ax=axs[1])
    xmin, xmax= math.floor(min([min(x_1),min(x_1)])),math.ceil(max([max(x_1),max(x_1)]))
    step = round((xmax-xmin)/steps)
    axs[0].set_xticks(range(xmin,xmax,step)), axs[1].set_xticks(range(xmin,xmax,step))
    axs[0].set_xticklabels(range(xmin,xmax,step),rotation=30), axs[1].set_xticklabels(range(xmin,xmax,step),rotation=30)
    axs[0].grid(), axs[1].grid()
    fig.tight_layout()
    fig.suptitle(file_name_1[0:-7]+'_degradation')
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_1[0:-7]+'_degradation'+'.png')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def A2P_HIGH(file_1,file_2,folder):
    print("******************************************** ANALYSIS A2P_HIGH ********************************************")
    print('interesting: passive sens, whips again')
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_1216\\A2P_HIGH'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    LET = LET_70_Si
    Flux = A2[1]
    dose_rate = LET*Flux*gy_conv # Gy/s
    # approx: beam starts at t=3e3 (i=), stops at t=555e3 (i=). S2 becomes active at t=212078.9766
    i_beam_start_1, i_beam_end_1 = dataset_1.index[dataset_1[t_info] >= 3e3].tolist()[0], dataset_1.index[dataset_1[t_info] >= 555e3].tolist()[0]
    i_recharge_start_1, i_recharge_end_1 = dataset_1.index[dataset_1[R_r_info]>0].tolist()[0], dataset_1.index[dataset_1[R_r_info] == 1].tolist()[0]
    i_active_1, i_active_2 = dataset_1.index[dataset_1[t_info] <= 212078].tolist()[-1], 1
    i_beam_start_2, i_beam_end_2 = 0, dataset_2.index[dataset_2[t_info] >= 555e3].tolist()[0]

    # plot temperature to see of compensation might be needed (together with reference frequency to see influence)
    steps = 15
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y)
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'])

    # plot overall view
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    x = [x_1,x_1,x_2,x_2]
    y = [y_1_s, y_1_r, y_2_s, y_2_r]
    names = ['F sensor 1','F reference 1','F sensor 2','F reference 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Fs&r',names,steps,savefolder=savefolder,colors=['red','darkred','blue','darkblue'],
                    markers=['o','.','o','.'])

    # use end and startpoints to determine sensitivity during passive period. Use recharge of Sensor 1 to determine t_recharge.
    t_start_passive_2, t_end_passive_2 = dataset_2[t_info][i_beam_start_2], dataset_2[t_info][i_active_2]
    dose_2_passive = dose_rate*(t_end_passive_2-t_start_passive_2)/1000
    dF_2 = dataset_2[F_s_info][i_beam_start_2] - dataset_2[F_s_info][i_active_2]
    sens_A2P_HIGH = (dF_2)/dose_2_passive
    print("sensitivity over passive range (endpoints) (S2) (Hz/Gy): %f"%(sens_A2P_HIGH)) 
    t_recharge_start, t_recharge_end = dataset_1[t_info][i_recharge_start_1], dataset_1[t_info][i_recharge_end_1]
    t_recharge_1 = (t_recharge_end-t_recharge_start)/1000
    print("time for a recharge (S1) (s): %f"%(t_recharge_1))
    print("charge lost during recharge (S1) (Gy): %f"%(t_recharge_1*dose_rate))

    # # determine sensitivity via linreg over linear range only (dataset split in active and passive parts, whips removed)
    # for the max dF for the whips, S1 is taken as a reference. The linear range is assumed to be extendable
    dF_1 = max_change(dataset_1,0,i_active_1)
    dataset_1_new = drop_shootouts(dataset_1,i_beam_start_1,i_recharge_start_1,dF_1)
    dataset_2_new = drop_shootouts(dataset_2,i_active_2,i_beam_end_2,dF_1)
    index_ranges =  [[i_beam_start_1,i_active_1,i_beam_start_2,i_active_2],[i_active_1,i_recharge_start_1,i_active_2,i_beam_end_2]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over divided range (linreg without whips) (each sensor is a row) (S1 || S2) (Hz/Gy): ")
    print(sensitivities)
    print("p-values of linreg fits for intercept and coeff (S1 || S2): ")
    print(pvalues)

    # plot linreg sensitivities
    dataset_2_new = pd.concat([dataset_2.loc[[i_beam_start_2]],dataset_2_new])
    y_2_s_new = dataset_2_new[F_s_info]
    x_2_new = dataset_2_new[t_info]
    xmin, xmax = limits([x_2_new])
    step = round((xmax-xmin)/steps)
    x_linreg = range(xmin,xmax,500)

    # fig, ax = plt.subplots(figsize=(15,7))
    # sb.scatterplot(x=x_1_new, y=y_1_s_new,color='salmon',label = 'Sensor 1',ax=ax)
    # sb.lineplot(x=x_linreg, y=f_linreg(coeffs[0][0],intercepts[0][0],x_linreg),color='royalblue', label = str(int(sensitivities[0][0]))+' (Hz/Gy)',ax=ax)
    # sb.lineplot(x=x_linreg, y=f_linreg(coeffs[0][1],intercepts[0][1],x_linreg),color='darkblue', label = str(int(sensitivities[0][1]))+' (Hz/Gy)',ax=ax)
    # ax.set_xticks(range(xmin,xmax,step))
    # ax.set_xticklabels(range(xmin,xmax,step),rotation=30)
    # ax.grid()
    # fig.suptitle(file_name_1[0:-7]+'_s1_linreg')
    # fig.show()
    # fig.savefig(savefolder+'\\'+file_name_1[0:-7]+'_s1_linreg'+'.png')

    fig, ax = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_2_new, y=y_2_s_new,color='mediumslateblue',label = 'Sensor 2',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][0],intercepts[1][0],x_linreg),color='indianred', label = str(int(sensitivities[1][0]))+' (Hz/Gy)',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][1],intercepts[1][1],x_linreg),color='darkred', label = str(int(sensitivities[1][1]))+' (Hz/Gy)',ax=ax)
    ax.set_xticks(range(xmin,xmax,step))
    ax.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax.grid()
    # fig.suptitle(file_name_1[0:-7]+'_s2_linreg')
    fig.tight_layout()
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_1[0:-7]+'_s2_linreg'+'.png')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def A2P_MANUAL_HIGH(file_1,file_2,folder):
    print("******************************************** ANALYSIS A2P_MANUAL_HIGH ********************************************")
    print('interesting: passive sens, extreme values when switching')
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_1216\\A2P_MANUAL_HIGH'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    LET = LET_70_Si
    Flux = A2[1]
    dose_rate = LET*Flux*gy_conv # Gy/s
    # approx: beam starts at t= (i=), stops at t= (i=).

    # drop extreme values
    dataset_2[F_s_info] = dataset_2[F_s_info].loc[(dataset_2[F_s_info]<=100000) & (dataset_2[F_s_info]>=40000)]
    dataset_2[F_r_info] = dataset_2[F_r_info].loc[(dataset_2[F_r_info]<=100000) & (dataset_2[F_r_info]>=65000)]
    
    # plot temperature to see of compensation might be needed (together with reference frequency to see influence)
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y)
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'])

    # plot overall view
    steps = 30
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    x = [x_1,x_1,x_2,x_2]
    y = [y_1_s, y_1_r, y_2_s, y_2_r]
    names = ['F sensor 1','F reference 1','F sensor 2','F reference 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Fs&r',names,steps,savefolder=savefolder,colors=['red','darkred','blue','darkblue'],
                    markers=['o','.','o','.'])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def A3P_MANUAL_HIGH(file_1,file_2,folder):
    print("******************************************** ANALYSIS A3P_MANUAL_HIGH ********************************************")
    print('interesting: passive sens, extreme values when switching')
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_1216\\A3P_MANUAL_HIGH'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    LET = LET_70_Si
    Flux = A3[1]
    dose_rate = LET*Flux*gy_conv # Gy/s
    # approx: beam starts at t= (i=), stops at t= (i=).

    # drop extreme values
    dataset_2[F_s_info] = dataset_2[F_s_info].loc[(dataset_2[F_s_info]<=90000) & (dataset_2[F_s_info]>=20000)]
    dataset_2[F_r_info] = dataset_2[F_r_info].loc[(dataset_2[F_r_info]<=100000) & (dataset_2[F_r_info]>=65000)]
    
    # plot temperature to see of compensation might be needed (together with reference frequency to see influence)
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y)
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'])

    # plot overall view
    steps = 30
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    x = [x_1,x_1,x_2,x_2]
    y = [y_1_s, y_1_r, y_2_s, y_2_r]
    names = ['F sensor 1','F reference 1','F sensor 2','F reference 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Fs&r',names,steps,savefolder=savefolder,colors=['red','darkred','blue','darkblue'],
                    markers=['o','.','o','.'])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def A3P_MANUAL_LOW(file_1,file_2,folder):
    print("******************************************** ANALYSIS A3P_MANUAL_LOW ********************************************")
    print('interesting: passive sens, extreme values when switching')
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_1216\\A3P_MANUAL_LOW'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    LET = LET_70_Si
    Flux = A3[1]
    dose_rate = LET*Flux*gy_conv # Gy/s
    # approx: beam starts at t= (i=), stops at t= (i=).

    # drop extreme values
    dataset_2[F_s_info] = dataset_2[F_s_info].loc[(dataset_2[F_s_info]<=250000) & (dataset_2[F_s_info]>=160000)]
    dataset_2[F_r_info] = dataset_2[F_r_info].loc[(dataset_2[F_r_info]<=190000) & (dataset_2[F_r_info]>=170000)]
    
    # plot temperature to see of compensation might be needed (together with reference frequency to see influence)
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y)
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'])

    # plot overall view
    steps = 30
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    x = [x_1,x_1,x_2,x_2]
    y = [y_1_s, y_1_r, y_2_s, y_2_r]
    names = ['F sensor 1','F reference 1','F sensor 2','F reference 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Fs&r',names,steps,savefolder=savefolder,colors=['red','darkred','blue','darkblue'],
                    markers=['o','.','o','.'])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def A2S_HIGH(file_1,file_2,folder):
    print("******************************************** ANALYSIS A2S_HIGH ********************************************")
    print('interesting: sens after lots of TID')
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_1216\\A2S_HIGH'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    LET = LET_70_Si
    Flux = A3[1]
    dose_rate = LET*Flux*gy_conv # Gy/s
    # approx: beam starts at t=4e3 (i=), stops at t=6e5 (i=). Start weird behaviour S1 ref and recharge reg t=499728.9062
    # due to the strange behaviour, start and end of recharge for S1 are the same
    i_beam_start_1, i_beam_end_1 = dataset_1.index[dataset_1[t_info] >= 4e3].tolist()[0], dataset_1.index[dataset_1[t_info] >= 6e5].tolist()[0]
    i_recharge_start_1 = dataset_1.index[dataset_1[R_r_info]>=128].tolist()[0] 
    i_recharge_end_1 = i_recharge_start_1+1
    i_error_start_1, i_error_start_2 = dataset_1.index[dataset_1[t_info] >= 499728].tolist()[0], dataset_2.index.tolist()[-1]
    i_beam_start_2, i_beam_end_2 = dataset_2.index[dataset_2[t_info] >= 4e3].tolist()[0], i_error_start_2
    i_recharge_start_2, i_recharge_end_2 = dataset_2.index[dataset_2[R_r_info]>=128].tolist()[0], dataset_2.index[dataset_2[R_r_info] == 1].tolist()[0]
    
    # plot temperature to see of compensation might be needed (together with reference frequency to see influence)
    steps = 30
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits(x)
    ymin, ymax = limits(y)
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    y = [y_1_r,y_2_r]
    ymin, ymax = limits(y) # limits are off due to error
    names = ['F reference 1','F reference 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'])

    # plot overall view
    steps = 30
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    x = [x_1,x_1,x_2,x_2]
    y = [y_1_s, y_1_r, y_2_s, y_2_r]
    names = ['F sensor 1','F reference 1','F sensor 2','F reference 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Fs&r',names,steps,savefolder=savefolder,ylim=[30000,100000],colors=['red','darkred','blue','darkblue'],
                    markers=['o','.','o','.'])


    # determine sensitivity via linreg over linear range only (dataset is modified to exlude RF trip, and split because of difference in linear response)
    index_ranges = [[i_beam_start_1,i_error_start_1,i_beam_start_2,i_recharge_start_2]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over linear range (linreg) (S1a | S1b || S2a | S2b) (Hz/Gy): ")
    print(sensitivities)
    print("p-values of linreg fits for intercept and coeff (S1a | S1b | S2a | S2b) (Hz/Gy): ")
    print(pvalues)

    # plot linreg sensitivities
    y_1_s, y_2_s = [dataset_1[F_s_info],dataset_2[F_s_info]]
    step = round((xmax-xmin)/steps)
    x_linreg = range(xmin,xmax,500)

    fig, ax = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_1, y=y_1_s,color='salmon',label = 'Sensor 1',ax=ax)
    sb.scatterplot(x=x_2, y=y_2_s,color='mediumslateblue',label = 'Sensor 2',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[0][0],intercepts[0][0],x_linreg),color='red', label = str(int(sensitivities[0][0]))+' (Hz/Gy)',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][0],intercepts[1][0],x_linreg),color='blue', label = str(int(sensitivities[1][0]))+' (Hz/Gy)',ax=ax)
    ax.set_xticks(range(xmin,xmax,step))
    ax.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax.grid()
    fig.suptitle(file_name_1[0:-7]+'_linreg')
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_1[0:-7]+'_linreg'+'.png')

    # determine sensitivities at the last time stamp. Delete whips for S2 first, linreg on small part, try to use sens from previous part?
    dF = max_change(dataset_2,i_beam_start_2,i_recharge_start_2)
    dataset_2_new = drop_shootouts(dataset_2,i_recharge_end_2,-1,dF)
    x_1, x_2 = dataset_1.loc[i_recharge_end_1+15:i_beam_end_1,t_info], dataset_2_new[t_info].iloc[10:-1]
    y_1, y_2 = dataset_1.loc[i_recharge_end_1+15:i_beam_end_1,F_s_info], dataset_2_new[F_s_info].iloc[10:-1]
    index_ranges = [[(i_recharge_end_1+15),i_beam_end_1,dataset_2_new.index.tolist()[0],dataset_2_new.index.tolist()[-1]]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2_new])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy (coeffs in Hz/s)
    print("sensitivity over second linear range (linreg) (S1 || S2) (Hz/Gy): ")
    print(sensitivities)
    # plot linreg sensitivities
    y_1_s, y_2_s = [y_1,y_2]
    xmin, xmax = limits([x_1, x_2])
    step = round((xmax-xmin)/steps)
    x_linreg = range(xmin,xmax,500)
    fig, ax = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_1, y=y_1_s,color='salmon',label = 'Sensor 1',ax=ax)
    sb.scatterplot(x=x_2, y=y_2_s,color='mediumslateblue',label = 'Sensor 2',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[0][0],intercepts[0][0],x_linreg),color='red', label = str(int(sensitivities[0][0]))+' (Hz/Gy)',ax=ax)
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][0],intercepts[1][0],x_linreg),color='blue', label = str(int(sensitivities[1][0]))+' (Hz/Gy)',ax=ax)
    ax.set_xticks(range(xmin,xmax,step))
    ax.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax.grid()
    fig.suptitle(file_name_1[0:-7]+'_linreg_secondrange')
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_1[0:-7]+'_linreg_secondrange'+'.png')

    print("endpoint sensor frequencies (Hz) (S1 | S2): %f | %f"%(f_linreg(coeffs[0][0],intercepts[0][0],xmax),f_linreg(coeffs[1][0],intercepts[1][0],xmax)))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def HIGH_B3_Feb_24_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS HIGH_B3_Feb_24 ********************************************")
    print("interesting: change in sensitivity with tid, recharges, annealing (file contains 2 test runs!)")
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_0224\\B3_HIGH'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dose_rate = 0.007567424 # Gy/s, see calculation table in excel, approx 120 MeV at target

    # use recharge reg and frequencies to determine endpoints of linear region
    # start lin region a bit after a recharge ends to exclude overshoots. Second test run stops at t = 84e4 ms
    i_start_1_1, i_end_1_1 = dataset_1.index[dataset_1[F_s_info] <= 90000].tolist()[0], dataset_1.index[dataset_1[F_s_info] <= 50000].tolist()[0] 
    i_start_1_2, i_end_1_2 = dataset_1.index[dataset_1[R_r_info] == 1].tolist()[5], dataset_1.index[dataset_1[R_r_info] == 129].tolist()[0]
    i_start_1_3, i_end_1_3 = dataset_1.index[dataset_1[R_r_info] == 2].tolist()[5], dataset_1.index[(dataset_1[t_info] >= 84e4)].tolist()[0]
    i_start_1_4, i_end_1_4 = dataset_1.index[dataset_1[R_r_info] == 3].tolist()[5], dataset_1.index[dataset_1[R_r_info] == 131].tolist()[0]
    i_start_1_5, i_end_1_5 = dataset_1.index[dataset_1[R_r_info] == 4].tolist()[5], dataset_1.index[dataset_1[R_r_info] == 132].tolist()[0]
    i_start_2_1, i_end_2_1 = dataset_2.index[dataset_2[F_s_info] <= 90000].tolist()[0], dataset_2.index[dataset_2[F_s_info] <= 50000].tolist()[0] 
    i_start_2_2, i_end_2_2 = dataset_2.index[dataset_2[R_r_info] == 1].tolist()[5], dataset_2.index[dataset_2[R_r_info] == 129].tolist()[0]
    i_start_2_3, i_end_2_3 = dataset_2.index[dataset_2[R_r_info] == 2].tolist()[5], dataset_2.index[(dataset_2[t_info] >= 84e4)].tolist()[0]
    i_start_2_4, i_end_2_4 = dataset_2.index[dataset_2[R_r_info] == 3].tolist()[5], dataset_2.index[dataset_2[R_r_info] == 131].tolist()[0]
    i_start_2_5, i_end_2_5 = dataset_2.index[dataset_2[R_r_info] == 4].tolist()[5], dataset_2.index[dataset_2[R_r_info] == 132].tolist()[0]

    # plot temperature to see of compensation might be needed. Also plot reference to see change over there (1 reading is of so ymax is changed)
    steps = 15
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits([x_1,x_2])
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    names = ['F reference sensor 1', 'F reference sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'])

    # linear range sensitivities using linear regression (in three parts)
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[F_s_info],dataset_2[F_s_info]
    index_ranges =  [[i_start_1_1,i_end_1_1,i_start_2_1,i_end_2_1],
                    [i_start_1_2,i_end_1_2,i_start_2_2,i_end_2_2],
                    [i_start_1_3,i_end_1_3,i_start_2_3,i_end_2_3],
                    [i_start_1_4,i_end_1_4,i_start_2_4,i_end_2_4],
                    [i_start_1_5,i_end_1_5,i_start_2_5,i_end_2_5]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over linear range (linreg, 5 recharges) (S1) (Hz/Gy): ")
    print(sensitivities[0])
    print("sensitivity over linear range (linreg, 5 recharges) (S2) (Hz/Gy): ")
    print(sensitivities[1])
    print("p-values of linreg fits for intercept and coeff (S1): ")
    print(pvalues[0])
    print("R-squared of linreg fits for intercept and coeff (S1): ")
    print(rsquared[0])
    print("p-values of linreg fits for intercept and coeff (S2): ")
    print(pvalues[1])
    print("R-squared of linreg fits for intercept and coeff (S2): ")
    print(rsquared[1])

    # plot only tpart of the graph
    x_1,x_2 = dataset_1.loc[i_start_1_1:i_end_1_5,t_info],dataset_2.loc[i_start_2_1:i_end_2_5,t_info]
    y_1, y_2 = dataset_1.loc[i_start_1_1:i_end_1_5,F_s_info],dataset_2.loc[i_start_2_1:i_end_2_5,F_s_info]
    xmin, xmax = limits([x_1,x_2])
    fig, ax = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_1, y=y_1,color='indianred',label = 'Sensor 1',ax=ax, alpha=0.5,marker='o')
    sb.scatterplot(x=x_2, y=y_2,color='royalblue',label = 'Sensor 2',ax=ax, alpha=0.5,marker='o')

    step = round((xmax-xmin)/steps)
    x_linreg = range(xmin,xmax,500)
    #  linreg lines for S2
    # sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][0],intercepts[1][0],x_linreg),color='red', label = str(int(sensitivities[1][0]))+' (Hz/Gy)',ax=ax)
    # sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][1],intercepts[1][1],x_linreg),color='red', label = str(int(sensitivities[1][1]))+' (Hz/Gy)',ax=ax)
    # sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][2],intercepts[1][2],x_linreg),color='red', label = str(int(sensitivities[1][2]))+' (Hz/Gy)',ax=ax)
    # sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][3],intercepts[1][3],x_linreg),color='red', label = str(int(sensitivities[1][3]))+' (Hz/Gy)',ax=ax)
    # sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][4],intercepts[1][4],x_linreg),color='red', label = str(int(sensitivities[1][4]))+' (Hz/Gy)',ax=ax)
    # ax.set_xticks(range(xmin,xmax,step))
    # ax.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax.grid()
    ax.set_ylim([45000,90000])
    fig.tight_layout()
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_2[0:-7]+'_linregs_'+'.png')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def HIGH_D4_2_Feb_24_analysis(file_1,file_2,folder):
    print("******************************************** ANALYSIS D4_2_HIGH ********************************************")
    print("interesting: wrange of 230k to 10k Hz")
    savefolder = os.getcwd()+'\\Figures\\HollandPTC_0224\\D4_2_HIGH'
    file_name_1, file_name_2 = str(file_1), str(file_2)
    dataset_1, dataset_2 = read_file(folder,file_1),read_file(folder,file_2)
    dose_rate = 0.027886514 # Gy/s, see excel, approximation for 140 MeV at target

    # use end and startpoints. use recharge of Sensor 2 to determine t_recharge
    # approx: beam starts at t=4e4 ms 
    i_start_1, i_start_2 = dataset_1.index[dataset_1[t_info] >= 4e4].tolist()[0], dataset_2.index[dataset_2[t_info] >= 4e4].tolist()[0]
    i_start_1_1, i_end_1_1, = dataset_1.index[dataset_1[F_s_info] <= 90000].tolist()[0] , dataset_1.index[dataset_1[F_s_info] <=50000].tolist()[0]
    i_start_2_1, i_end_2_1 = dataset_2.index[dataset_2[F_s_info] <= 90000].tolist()[0] , dataset_2.index[dataset_2[F_s_info] <=50000].tolist()[0]

    # plot temperature to see of compensation might be needed. Also plot reference to see change over there (1 reading is of so ymax is changed)
    steps = 15
    x_1,x_2,y_1_T,y_2_T = dataset_1[t_info],dataset_2[t_info],dataset_1[T_info],dataset_2[T_info]
    x = [x_1,x_2]
    y = [y_1_T,y_2_T]
    xmin, xmax = limits([x_1,x_2])
    names = ['T sensor 1','T sensor 2']
    fig,ax = plot_xy(x,y,file_name_1[0:-7]+'_Temperature',names,steps,savefolder=savefolder,xlim=[0,xmax],ylim=[75,85],colors=['darkred','darkblue'])
    y_1_r,y_2_r = dataset_1[F_r_info],dataset_2[F_r_info]
    names = ['F reference sensor 1', 'F reference sensor 2']
    fig,ax = plot_2subs_datasets(x_1,x_2,y_1_r,y_2_r,file_name_1[0:-7]+'_Fr',names,steps,savefolder=savefolder,colors=['darkred','darkblue'])

    # linear range sensitivities using linear regression (in three parts)
    offset_test = 0 # offset can be used to test, set to 0 for normal linreg
    x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[F_s_info],dataset_2[F_s_info]
    index_ranges =  [[i_start_1_1,i_end_1_1,i_start_2_1,i_end_2_1]]
    intercepts, coeffs, pvalues, rsquared = sens_linreg(t_info,F_s_info,index_ranges,[dataset_1,dataset_2])
    sensitivities = -coeffs/dose_rate*1000 # Hz/Gy
    print("sensitivity over linear range (linreg) (S1 || S2) (Hz/Gy): ")
    print(sensitivities)
    print("p-values of linreg fits for intercept and coeff (S1 || S2): ")
    print(pvalues)
    print("R-squared of linreg fits for intercept and coeff (S1 || S2): ")
    print(rsquared)

    # plot with linreg
    x_1,x_2 = dataset_1.loc[i_start_1:dataset_1.index[-1],t_info],dataset_2.loc[i_start_2:dataset_2.index[-1],t_info]
    y_1, y_2 = dataset_1.loc[i_start_1:dataset_1.index[-1],F_s_info],dataset_2.loc[i_start_2:dataset_2.index[-1],F_s_info]
    xmin, xmax = limits([x_1,x_2])
    x_linreg = range(xmin,xmax,500)
    fig, ax = plt.subplots(figsize=(15,7))
    sb.scatterplot(x=x_2, y=y_2,color='royalblue',label = 'Sensor 2',ax=ax, alpha=0.5,marker='o')
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[1][0],intercepts[1][0],x_linreg),color='darkred', label = str(int(sensitivities[1][0]))+' (Hz/Gy)',ax=ax)
    sb.scatterplot(x=x_1, y=y_1,color='indianred',label = 'Sensor 1',ax=ax, alpha=0.5,marker='o')
    sb.lineplot(x=x_linreg, y=f_linreg(coeffs[0][0],intercepts[0][0],x_linreg),color='darkblue', label = str(int(sensitivities[0][0]))+' (Hz/Gy)',ax=ax)
    # ax.set_xticks(range(xmin,xmax,step))
    # ax.set_xticklabels(range(xmin,xmax,step),rotation=30)
    ax.grid()
    ax.set_ylim([0,240000])
    # fig.suptitle(file_name_2[0:-7]+'_linreg_extended')
    fig.tight_layout()
    fig.show()
    fig.savefig(savefolder+'\\'+file_name_2[0:-7]+'_linreg_extended'+'.png')