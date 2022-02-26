"""modify the excel files stored during the test
the function splits the data in 2 files (sensor 1 and 2) and stores it in an easy format for Pandas
"""

from FGD_03F_functions import *

# which files to analyse
start_of_filename = "FGDOS_03F"
folder = "CSV_files\\Arduino\\HollandPTC_0224"

def main():
    # get csv files in subfolder, set the start of filename parameter to select specific files
    files = [f for f in os.listdir(os.getcwd()+"\\"+folder) if (f.endswith(".csv") and f.startswith(start_of_filename))]
    file_names = [0]*len(files)
    
    j = 0
    for file in files:
        file_names[j] = modify_file(folder,file,columns_string)
        j += 1

    input("<Hit enter to close>\n")

if __name__ == "__main__":
    main()