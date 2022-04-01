"""analyse each file separately
instead of plotting,... over all the files, specific adaptations are made on a per file basis 
this allows for selection of the most important test results
"""

# TO ADD
# - compare dose rates with HollandPTC data! Verify HollandPTC dose rates....

from FGD_03F_functions import *
from FGD_03F_file_analysis_functions import *
from FGD_03F_noise_analysis_functions import *

flag_HollandPTC_1216 = 0
flag_HollandPTC_0224 = 0
flag_Noise = 1

if (flag_HollandPTC_1216):
    # which files to analyse
    start_of_filename = "FGDOS"
    folder = "CSV_files\\Arduino\\HollandPTC_1216\\corrected"
    # file flags
    flag_A1_HIGH2 = 0
    flag_A2_HIGH = 0
    flag_A2_LOW = 0
    flag_A3_HIGH = 0
    flag_A3_LOW = 0
    flag_A4_HIGH = 1
    flag_A4_LOW = 0
    flag_A2P_HIGH = 0
    flag_A2P_MANUAL_HIGH = 0
    flag_A3P_MANUAL_HIGH = 0
    flag_A3P_MANUAL_LOW = 0
    flag_A2S_HIGH = 0
elif (flag_Noise):
    # which files to analyse
    start_of_filename = "FGDOS"
    folder = "CSV_files\\Arduino\\Noise\\corrected"
    # file flags
    flag_Noise_01_HIGH_Dec_15 = 0
    flag_Noise_02_HIGH_Dec_15 = 0
    flag_Noise_03_LOW_Dec_15 = 0
    flag_Noise_04_HIGH_Dec_16 = 0
    flag_HIGH_Jan_14 = 0
    flag_HIGH_Jan_15 = 0
    flag_LOW_Jan_15 = 0
    flag_HIGH_Jan_26 = 0
    flag_HIGH_Feb_18 = 0
    flag_HIGH_w32768_Feb_22 = 0
    flag_HIGH_Feb_23 = 0
    flag_HIGH_w32768_Feb_23 = 1
elif (flag_HollandPTC_0224):
    # which files to analyse
    start_of_filename = "FGDOS"
    folder = "CSV_files\\Arduino\\HollandPTC_0224\\corrected"
    # file flags
    flag_HIGH_B3_Feb_24 = 1
    flag_HIGH_D4_2_Feb_24 = 0

def main():
    # get csv files in subfolder, set the start of filename parameter to select specific files
    files_1 = [f for f in os.listdir(os.getcwd()+"\\"+folder) if (f.endswith("s1.csv") and f.startswith(start_of_filename))]
    files_2 = [f for f in os.listdir(os.getcwd()+"\\"+folder) if (f.endswith("s2.csv") and f.startswith(start_of_filename))]

    if (flag_HollandPTC_1216):
        for file_1 in files_1:
            file_name_1 = str(file_1)
            for file_2 in files_2:
                file_name_2 = str(file_2)
                if (file_name_1[:-7] == file_name_2[:-7]):
                    if('A1_HIGH2' in file_name_1 and flag_A1_HIGH2):
                        A1_HIGH2_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('A2_HIGH' in file_name_1 and flag_A2_HIGH):
                        A2_HIGH_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('A2_LOW' in file_name_1 and flag_A2_LOW):
                        A2_LOW_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if ('A3_HIGH' in file_name_1 and flag_A3_HIGH):
                        A3_HIGH_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if ('A3_LOW' in file_name_1 and flag_A3_LOW):
                        A3_LOW_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if ('A4_HIGH' in file_name_1 and flag_A4_HIGH):
                        A4_HIGH_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if ('A4_LOW' in file_name_1 and flag_A4_LOW):
                        A4_LOW_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if ('A2P_HIGH_Dec_16_21-13-49' in file_name_1 and flag_A2P_HIGH):
                        # longer filename to discard other short file
                        A2P_HIGH(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if ('A2P_MANUAL_HIGH' in file_name_1 and flag_A2P_MANUAL_HIGH):
                        A2P_MANUAL_HIGH(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if ('A3P_MANUAL_HIGH' in file_name_1 and flag_A3P_MANUAL_HIGH):
                        A3P_MANUAL_HIGH(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if ('A3P_MANUAL_LOW' in file_name_1 and flag_A3P_MANUAL_LOW):
                        A3P_MANUAL_LOW(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if ('A2S_HIGH' in file_name_1 and flag_A2S_HIGH):
                        A2S_HIGH(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                else:
                    pass
    elif (flag_Noise):
        for file_1 in files_1:
            file_name_1 = str(file_1)
            for file_2 in files_2:
                file_name_2 = str(file_2)
                if (file_name_1[:-7] == file_name_2[:-7]):
                    if('Noise_01_HIGH_Dec_15' in file_name_1 and flag_Noise_01_HIGH_Dec_15):
                        Noise_01_HIGH_Dec_15_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('Noise_02_HIGH_Dec_15' in file_name_1 and flag_Noise_02_HIGH_Dec_15):
                        Noise_02_HIGH_Dec_15_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('Noise_03_LOW_Dec_15' in file_name_1 and flag_Noise_03_LOW_Dec_15):
                        Noise_03_LOW_Dec_15_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('Noise_04_HIGH_Dec_16' in file_name_1 and flag_Noise_04_HIGH_Dec_16):
                        Noise_04_HIGH_Dec_16_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('HIGH_Jan_14' in file_name_1 and flag_HIGH_Jan_14):
                        HIGH_Jan_14_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('HIGH_Jan_15' in file_name_1 and flag_HIGH_Jan_15):
                        HIGH_Jan_15_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('LOW_Jan_15_14-37-09' in file_name_1 and flag_LOW_Jan_15):
                        LOW_Jan_15_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('HIGH_Jan_26' in file_name_1 and flag_HIGH_Jan_26):
                        HIGH_Jan_26_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('HIGH_Feb_18' in file_name_1 and flag_HIGH_Feb_18):
                        HIGH_Feb_18_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('HIGH_w32768_Feb_22' in file_name_1 and flag_HIGH_w32768_Feb_22):
                        HIGH_w32768_Feb_22_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('HIGH_Feb_23' in file_name_1 and flag_HIGH_Feb_23):
                        HIGH_Feb_23_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('HIGH_w32768_Feb_23' in file_name_1 and flag_HIGH_w32768_Feb_23):
                        HIGH_w32768_Feb_23_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                else:
                    pass
    elif (flag_HollandPTC_0224):
        for file_1 in files_1:
            file_name_1 = str(file_1)
            for file_2 in files_2:
                file_name_2 = str(file_2)
                if (file_name_1[:-7] == file_name_2[:-7]):
                    if('HIGH_B3_Feb_24' in file_name_1 and flag_HIGH_B3_Feb_24):
                        HIGH_B3_Feb_24_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                    if('HIGH_D4.2_Feb_24' in file_name_1 and flag_HIGH_D4_2_Feb_24):
                        HIGH_D4_2_Feb_24_analysis(file_1,file_2,folder)
                        input("\n<Hit enter to continue>\n")
                else:
                    pass
    input("\n<Hit enter to close>\n")
    plt.close('all')

if __name__ == "__main__":
    main()

