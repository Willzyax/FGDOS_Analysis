from SAM_functions import *
import time

# folder to use and file selection via start of filename
folder = "CSV_files\\SAMV71\\Noise"
start_of_filename = "SAMV71_HIGH_w32768_Feb_23"
# set number of lines to skip in reading voltages to limit plot and analysis data
lineskip = 1

def main():
    # get csv files in subfolder, set the start of filename parameter to select specific files
    files = [f for f in os.listdir(os.getcwd()+"\\"+folder) if (f.endswith(".csv") and f.startswith(start_of_filename))]
    file_names = [0]*len(files)
    filename_stats = folder+"\\summary_stats.csv"

    # file by file analysis
    # files are changed to exclude invalid lines and entries, summary statistics of all files are saved to a csv file
    with open(filename_stats,'w',newline='') as statsfile:
        statsfile.seek(0)
        j = 0
        for file in files:
            file_names[j], dataset = modify_file(folder,file,lineskip)
            statsfile.write((file_names[j][0])+'\n')
            data_stats = dataset.agg({
                    t:["min","max"],
                    v_shunt_1:pd_stats_list,
                    v_bus_1:pd_stats_list,
                    i_shunt_1:pd_stats_list,
                    p_1:pd_stats_list,
                    v_shunt_2:pd_stats_list,
                    v_bus_2:pd_stats_list,
                    i_shunt_2:pd_stats_list,
                    p_2:pd_stats_list
                    })
            data_stats.to_csv(statsfile)
            # input("<Hit enter to go to next file>\n")
            j += 1
    # print(file_names)
    # print(data_dict)
    
    input("<Hit enter to close>\n")

if __name__ == "__main__":
    main()