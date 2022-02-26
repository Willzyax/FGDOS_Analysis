
import os

start_of_filename = "FGDOS"
folder = "CSV_files\\Arduino\\HollandPTC_1216\\PuTTy"
topfolder = "CSV_files\\Arduino\\HollandPTC_1216"
files = [f for f in os.listdir(os.getcwd()+"\\"+folder) if (f.endswith(".csv") and f.startswith(start_of_filename) and not f.endswith("timeinc.csv"))]


t_12_first = 1000 # µs to go first time from s1 to s2 
t_12 = 243.835*10**3 # µs overall average to go from s1 to s2
t_21 = 19.290*10**3 # µs overall average to go from s2 to s1
t_2A = 175*10**3 # time to put sensor 2 to active
t_2O = 1*10**3 # time to turn sensor 2 on
t_2R = 1280*10**3 # time to reset sensor 2
t_2P = t_2A # time to put sensor 2 to passive
t_2S = t_2O # time to put sensor 2 to standby

for f in files:
    file_name = str(f)
    first_run = True
    second_run = True
    with  open(folder+"\\"+file_name, "r",newline='') as original:
        print("------------- MODIFY FILE -------------")
        print(file_name)
        with open(folder+'\\'+file_name[0:-4]+'_timeinc.csv',"w",newline='') as new:
            for line in original: 
                data_line = line.split(",")
                while (data_line[0].strip(' ') != 'Sensor' and first_run):
                    temp = next(original)
                    data_line = temp.split(',')
                    new.write('0,'+temp)
                if (first_run):
                    first_run = False
                elif (not first_run and second_run):
                    new.write('0,'+line)
                    t_temp = t_12_first
                    new.write(str(t_12_first)+','+next(original))
                    second_run = False
                elif (not first_run and not second_run and data_line[0].strip(' ') == '1'):
                    t_temp = t_temp + t_21
                    new.write(str(t_temp)+','+line)
                elif (not first_run and not second_run and data_line[0].strip(' ') == '2'):
                    t_temp = t_temp + t_12
                    new.write(str(t_temp)+','+line)
                elif(not first_run and not second_run and data_line[0].split(' ')[0] == 'PASSIVE'):
                    t_temp = t_temp + t_2P
                    new.write(str(t_temp)+','+line)
                elif(not first_run and not second_run and data_line[0].split(' ')[0] == 'ACTIVE'):
                    t_temp = t_temp + t_2A
                    new.write(str(t_temp)+','+line)
                elif(not first_run and not second_run and data_line[0].split(' ')[0] == 'STANDBY'):
                    t_temp = t_temp + t_2S
                    new.write(str(t_temp)+','+line)
                elif(not first_run and not second_run and data_line[0].split(' ')[0] == 'ON'):
                    t_temp = t_temp + t_2O
                    new.write(str(t_temp)+','+line)
                elif(not first_run and not second_run and data_line[0].split(' ')[0] == 'RESET:'):
                    t_temp = t_temp + t_2R
                    new.write(str(t_temp)+','+line)
                else:
                    print('unknown condition',end=" ")
                    new.write(str(t_temp)+','+line)
                    print(line)
            input('<enter to go to next file>')
                    