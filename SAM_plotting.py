from SAM_functions import *
import matplotlib.pyplot as plt

# folder to use and file selection via start of filename
folder = "CSV_files\\SAMV71\\Noise\\corrected"
folder_image = "Figures\\Noise\\_SAM"
start_of_filename = "SAMV71_HIGH_w32768_Feb_23"
# figure settings
steps = 10

def main():
    # get csv files in subfolder, set the start of filename parameter to select specific files
    files = [f for f in os.listdir(os.getcwd()+"\\"+folder) if (f.endswith(".csv") and f.startswith(start_of_filename))]
    file_names = [0]*len(files)

    # file by file plot of powers, currents, bus voltages
    j = 0
    plt.ion()
    for file in files:
        file_names[j], dataset = read_file(folder,file)
        plot_data(str(os.getcwd()+"\\"+folder_image),dataset,file_names[j][0][0:-4]+'_Is',steps,[t,i_shunt_1,i_shunt_2])
        plot_data(str(os.getcwd()+"\\"+folder_image),dataset,file_names[j][0][0:-4]+'_Vb',steps,[t,v_bus_1,v_bus_2])
        plot_data(str(os.getcwd()+"\\"+folder_image),dataset,file_names[j][0]+'_P',steps,[t,p_1,p_2])
        
        ranges = hist_data(str(os.getcwd()+"\\"+folder_image),dataset,file_names[j][0][0:-4]+'_Vb_hist',bins=10,columns=[v_bus_1,v_bus_2],dist='none')
        print("Vb range (mV) (S1 | S2): %d | %d"%(ranges[0],ranges[1]))
        print("measurement time (s): %d"%((dataset[t].iloc[-1]-dataset[t].iloc[0])/1000))
        
        j += 1
        input("<Hit enter to go to next file>\n")
        plt.close('all')

    input("<Hit enter to close>\n")

if __name__ == "__main__":
    main()