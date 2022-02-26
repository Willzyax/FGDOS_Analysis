import serial
import time
import os

def main():
    # create datafile to store into
    date_time = time.strftime("%b_%d_%H-%M-%S", time.localtime())
    datafile = "CSV_files\\FGDOS_02F_" + date_time + ".csv"
    
    # connect with Arduino
    arduino_port = "COM4"  # serial port of Arduino
    baud = 250000  # set at same rate as in Arduino program
    ser = serial.Serial(arduino_port, baud, timeout=0.5, write_timeout = 0.5)
    print("Connected to Arduino port: " + arduino_port)

    # Commands
    # - 1 or 2 to read sensors 1 or 2
    # - R to simple reset sensor 2
    # - A activate sensor 2
    # - D deactivate sensor 2 (disconnect from power via mosfet)
    # - tab enter exits

    # append the data to the file or create new, use raw string to prevent \ escapes
    file = open(datafile, "a")
    time_stamp = time.time_ns()
    command = ''
    while True:
        command = input("Command? ")
        while command == '': command = input("Command? ") 
        if command == '\t': break 
        ser.write(command.encode())
        ser.flush()
        time.sleep(0.1)
        
        start = time.time()
        while ser.in_waiting<1:
            time.sleep(0.05)
            if (time.time()-start > 3):
                break
        while ser.in_waiting>0:
            t = time.time_ns() - time_stamp
            data = ser.readline().decode("ASCII")[0:-1]
            datastring = str(t) + ',' + data
            print(datastring)
            file.write(datastring)

    file.close()
    print("------------------ END ------------------")

if __name__ == "__main__":
    main()