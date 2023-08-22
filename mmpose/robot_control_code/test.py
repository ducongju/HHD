import serial
import time
portx = 'COM1'
bps = 9600
timex = 5
ser = serial.Serial(portx, bps, timeout=timex)

time.sleep(1)
code = "EF FE 00 00 00 00 00 00 00 ED"
b = bytes.fromhex(code)
result = ser.write(b)
data = ser.read(10)
print(result)
print('data：'+str(data))
ser.close()
