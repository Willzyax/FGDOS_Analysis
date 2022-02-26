'''
same as read save 02F but without any live plots to prevent time loss
'''

import time
import serial

def main():
    # create datafile to store into
    date_time = time.strftime("%b_%d_%H-%M-%S", time.localtime())
    datafile = "02F_files\\CSV_files\\Arduino_Noise_" + date_time + ".csv"

    # connect with Arduino
    arduino_port = "COM9"  # serial port of Arduino
    baud = 250000  # set at same rate as in Arduino program
    ser = serial.Serial(arduino_port, baud, timeout=0.1)
    print("Connected to Arduino port: " + arduino_port)

    # display the data to the terminal and save it to csv file
    file = open(datafile, "a", encoding= "ASCII")
    time_stamp = time.time_ns()
    ser.flushInput()

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
                except Exception as e:
                    print("error reading serial")
                    print(e)
                    break
                
                datastring = str(t) + " , " + data
                print(datastring)
                file.write(datastring)
    except KeyboardInterrupt:
        pass

    file.close()
    print("---------------------- file closed, end of program ----------------------")

if __name__ == "__main__":
    main()
