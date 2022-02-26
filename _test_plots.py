# importing libraries
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sb
 
folder = "CSV_files\\Arduino\\HollandPTC_1216\\corrected"
file_name = 'FGDOS_03F_A1_HIGH_Dec_16_19-02-12_s1.csv'
file_folder = folder+'\\'+file_name

dataset = pd.read_csv(file_folder, skiprows = 0, delimiter = ",")
print(dataset)

# x = dataset["Time (ms)"]
# y = dataset["Sensor Frequency (Hz)"]
# z = dataset["Reference Frequency (Hz)"]
# fig, ax = plt.subplots(figsize=(15,7))
# sb.scatterplot(x=x, y=y, data=dataset,color='red').set(title=file_name)
# sb.scatterplot(x=x, y=z, data=dataset,color='blue').set(title=file_name)
# fig.canvas.draw()
# fig.canvas.flush_events()
# fig.show()

t_info = "Time (ms)"
F_s_info = "Sensor Frequency (Hz)"
F_s = dataset[F_s_info]
t_s = dataset[t_info]

F_start = np.mean(F_s.loc[0:100]) 
F_end = np.min(F_s)
t_start = t_s[F_s == F_s.iat[100]].iat[0]
t_end = t_s[F_s == F_end].iat[0]

print('%f,%f,%f,%f'%(F_start,F_end,t_start,t_end))
input("<Hit enter to close>\n")