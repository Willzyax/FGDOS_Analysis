# TO DO
# - make use of csv folder
# - make sure reading data and reading settings is done correctly
# - split data correctly
# - add different temperature compensations
# - plots? save to excel? (suggest to do plots in different script)

from time import sleep
import pandas as pd
import os

# get csv files in subfolder
# set string start_of_filename to select files by setting what name they should start with
start_of_filename = "FGDOS_02F_Oct_14"
folder = "CSV_files"
files = [f for f in os.listdir(os.getcwd()+"\\"+folder) if (f.endswith(".csv") and f.startswith(start_of_filename))]

for file in files:
    fileName = str(file)

    # first read the general setup settings
    f = open(folder+"\\"+fileName, "r")
    print("OPENED FILE: "+fileName)

    # read first line, check to see if it is the sensor number, if not keep on reading lines until sensor number encountered
    sensor_number = f.readline().split(",")[-1].replace(" ","")
    i = 0
    while sensor_number[0:6] != "SENSOR":
        print("skipped line, reading next one...")
        sensor_number = f.readline().split(" ")[-1]
        i = i+1
        if i>10:
            print("no appropriate start found in first 10 lines")
            break
    sensor_number = int(sensor_number[-2])
    window_factor = float(f.readline().split(" ")[-1][:-1])
    sensitivity = (
        5 if f.readline().split(" ")[1][0:3] == "low" else 30
    ) * 1000  # Hz/Gy
    f.close()
    print("sensor: ", sensor_number," window: ",window_factor)

    # read data into dataframe to perform calculations
    fgdos_data = pd.read_csv(fileName, comment="-")  # us row 3 as header
    # print(fgdos_data)
    columns = fgdos_data.columns  # delete spaces in columns names
    for old in columns:
        new = old.replace(" ", "")
        fgdos_data.rename(columns={old: new}, inplace=True)

    # difference between new and last with temperature compensation based on linear equation
    # Grey depends on sensitivity
    # sensor sensitivity to rad (?) not included
    fgdos_data["dFs_RT"] = 0
    fgdos_data["Gr"] = 0.0
    for i in fgdos_data.index[:-1]:
        fgdos_data.loc[i + 1, "dFs_RT"] = (
            fgdos_data.loc[i + 1, "Fr"] - fgdos_data.loc[i, "Fs"]
        )
        fgdos_data.loc[i + 1, "Gr"] = (fgdos_data.loc[i + 1, "dFs_RT"]) / sensitivity

    print(fgdos_data)
