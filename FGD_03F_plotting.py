"""plot the data for both sensor 1 and 2 for each separate test to get a quick overview of the results
"""

from FGD_03F_functions import *

# which files to analyse
start_of_filename = "FGDOS"
folder = "CSV_files\\Arduino\\HollandPTC_0224\\corrected"

# plot settings
steps = 15
savefolder = 'Figures\\HollandPTC_0224\\_General'

def main():
    # get csv files in subfolder, set the start of filename parameter to select specific files
    files_1 = [f for f in os.listdir(os.getcwd()+"\\"+folder) if (f.endswith("s1.csv") and f.startswith(start_of_filename))]
    files_2 = [f for f in os.listdir(os.getcwd()+"\\"+folder) if (f.endswith("s2.csv") and f.startswith(start_of_filename))]

    for file_1 in files_1:
        file_name_1 = str(file_1)
        for file_2 in files_2:
            file_name_2 = str(file_2)
            if (file_name_1[:-7] == file_name_2[:-7]):
                dataset_1 = read_file(folder,file_1)
                dataset_2 = read_file(folder,file_2)
                x_1,x_2,y_1,y_2 = dataset_1[t_info],dataset_2[t_info],dataset_1[F_s_info],dataset_2[F_s_info]

                plot_datasets(x_1,x_2,y_1,y_2,file_name_1[0:-7],steps,savefolder=savefolder)
            else:
                pass
          
        # input("<Hit enter to continue>\n")
    input("<Hit enter to close>\n")

if __name__ == "__main__":
    main()