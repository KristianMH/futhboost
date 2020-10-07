from array import array
floats = [3.14, 2.7, 0.0, -1.0, 1.1]
import struct
s = struct.pack('f'*len(floats), *floats)
f = open('dat','wb')
f.write(s)
f.close()
