import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial

def main():
    # create datafile to store into
    date_time = time.strftime("%b_%d_%H-%M-%S", time.localtime())
    datafile = "CSV_files\\FGDOS_02F_" + date_time + ".csv"

    # connect with Arduino
    arduino_port = "COM4"  # serial port of Arduino
    baud = 250000  # set at same rate as in Arduino program
    ser = serial.Serial(arduino_port, baud, timeout=0.1)
    print("Connected to Arduino port: " + arduino_port)

    # display the data to the terminal and save it to csv file
    # additionally, a matplotlib figure is created to show data in real time
    file = open(datafile, "a", encoding= "ASCII")
    time_stamp = time.time_ns()
    ser.flushInput()

    # Several options exist to plot live data (matplotlib https://matplotlib.org/2.0.2/faq/usage_faq.html)
    # - option 1
    #     redraw the complete plot everytime, also autoupdating the axis 
    #     but this takes too much time
    # - option 2
    #     only redraw the lines, which makes it much faster, but the axis stay fixed
    #     there are 2 ways to do this, 
    #       2.1 either using canvas.draw which should be faster https://matplotlib.org/devdocs/tutorials/advanced/blitting.html
    #       2.2 or by using funcanimation, in which everything in the loop should be placed in a function https://matplotlib.org/stable/api/animation_api.html#module-matplotlib.animation
    # additionally a counter is used to prevent redrawing every serial read loop
    x = []
    y_ref = []
    y_sens = []
    counter = 0
    
    # # option 1
    # plt.ion()
    # fig, (ax_1,ax_2) = plt.subplots(1,2,sharex=True,sharey=False,figsize=(10, 8))
    # line_ref, = ax_1.plot(x,y_ref,'r--')
    # line_sens, = ax_2.plot(x,y_sens,'b-')
    # ax_1.autoscale(enable=True,axis='both')
    # ax_2.autoscale(enable=True,axis='both')
    # ax_1.set(xlabel="Time (ns)",ylabel="Frequency (Hz)")
    # ax_2.set(xlabel="Time (ns)",ylabel="Frequency (Hz)")

    # option 2
    plt.ion()
    fig, (ax_1,ax_2) = plt.subplots(1,2,sharex=True,sharey=False,figsize=(10, 8))
    line_ref, = ax_1.plot(x,y_ref,'r--')
    line_sens, = ax_2.plot(x,y_sens,'b-')
    ax_1.set_xlim([0,100000])
    ax_1.set_ylim([40000,50000])
    ax_2.set_ylim([40000,50000])
    ax_1.set(xlabel="Time (ns)",ylabel="Frequency (Hz)")
    ax_2.set(xlabel="Time (ns)",ylabel="Frequency (Hz)")

    try:
        while True:
            # wait for data incoming over serial, as soon as something enters buffer, pass
            start = time.time()
            while ser.in_waiting < 1:
                if (time.time() - start) > 3:
                    print("timeout waiting for serial")
                    break
            while ser.in_waiting > 0:
                try:
                    data = ser.readline().decode("ASCII")[:-1]
                    data_array = data.split(",")
                    t = time.time_ns() - time_stamp
                except:
                    print("error reading serial")
                    break
                
                try:
                    if(int(data_array[0]) == 2):
                        y_sens.append(int(data_array[2]))
                        y_ref.append(int(data_array[3]))
                        x.append(int(t/1000000))
                        counter += 1
                except:
                    pass

                # draw after number of added data points
                if counter>3:  
                    # option 1 (reset complete figure everytime)
                    # ax_1.clear()
                    # ax_2.clear()
                    # line_ref, = ax_1.plot(x,y_ref,'r--')
                    # line_sens, = ax_2.plot(x,y_sens,'b-')
                    # plt.pause(0.5)

                    # option 2 (redraw only part of the figure)
                    line_ref.set_xdata(x)
                    line_ref.set_ydata(y_ref)
                    line_sens.set_data(x,y_sens)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    counter = 0
                
                datastring = str(t) + " , " + data
                print(datastring)
                file.write(datastring)

    except KeyboardInterrupt:
        pass

    file.close()
    print("---------------------- file closed, end of program ----------------------")


# option to use funcanimation, note that the entire loop above should be in the function, not sure of this is possible
# freq_animation = animation.FuncAnimation(fig,animate,fargs= (x,y_ref,y_sens), interval = 1000,blit = True, save_count = 50)
# def animate(i, x, y_ref,y_sens):
#     line_ref.set_data(x,y_ref)
#     line_sens.set_data(x,y_sens)
#     return line_ref,line_sens

if __name__ == "__main__":
    main()