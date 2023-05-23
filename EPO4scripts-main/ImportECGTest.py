import time
from pySerialTransfer import pySerialTransfer as txfer
import csv

def import_all(COMport = '/dev/ttyACM0'):
    print("Importing ECG, GSR and respiratory data \n")
    try:
        link = txfer.SerialTransfer(COMport, baud=57600)

        open_link = link.open()
        print(f"Link is open: {open_link}")

        header = ['TimeStamp', 'ECG Data', 'GSR Data']
        data = [0, 0, 0]

        output_file = open('ECGdata.csv', 'w')
        writer = csv.writer(output_file)
        writer.writerow(header)

        time.sleep(3)
        print(f"link status: {link.status}")
        print('Stop data collection by keyboard interrupt (ctrl+c)')
        
        while True:
            if link.available():
                recSize = 0

                # Import time stamp of the data sample
                data[0] = link.rx_obj(obj_type='L', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['L']

                # Import ECG data from serial connection
                data[1] = link.rx_obj(obj_type='H', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['H']

                # Import GSR data from serial connection
                data[2] = link.rx_obj(obj_type='H', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['H']

                writer.writerow(data)

            elif link.status < 0:
                if link.status == txfer.CRC_ERROR:
                    print('ERROR: CRC_ERROR')
                elif link.status == txfer.PAYLOAD_ERROR:
                    print('ERROR: PAYLOAD_ERROR')
                elif link.status == txfer.STOP_BYTE_ERROR:
                    print('ERROR: STOP_BYTE_ERROR')
                else:
                    print('ERROR: {}'.format(link.status))


    except KeyboardInterrupt:
        link.close()
        output_file.close()
        print("Serial connection & output_file closed. \n Don't forget to rename ECGdata to prevent overwriting!")

