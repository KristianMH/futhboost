import numpy as np
import futhark_data
import os

path = "histogram_bench_data/"
try:
    os.mkdir(path)
except:
    print("dir already exists")
    pass


prefix = "bench"
def path_name(path, prefix, n,k):
    return path+prefix+"_"+str(n)+"_"+str(k)

def file_name(prefix, n, k):
    return prefix+"_"+str(n)+"_"+str(k)
def matsize_to_str(n,k):
    ks = "["+str(k)+"]"
    ns = "["+str(n)+"]"
    return ns + ks

def size_to_str(n):
    ns = "["+str(n)+"]"
    return ns

datasets = os.listdir(path)

SEGS = np.array([2**4, 2**5, 2**6, 2**7, 2**8, 2**9]).astype("int64")
NS = [10**3, 10**4, 10**5, 10**6, 10**7, 10**7*2]

for (num, size) in zip(NS, SEGS):
    str_size = matsize_to_str(num, 20)
    arr_size = size_to_str(num)
    name = path_name(path, prefix, num, size)
    # #print(file_name(prefix, num, size) not in datasets)
    if file_name(prefix, num, size) not in datasets:
        #data = np.random.rand(num,size).astype("float32")
        #print(data.shape, data.dtype)
        rnd_shape = np.random.multinomial(num, np.ones(size)/size, size=1)[0].astype("int64")
        #print(rnd_shape.shape, rnd_shape.dtype)
        
        #print("making "+str_size+" matrix with name "+ name)
        print("Making: "+ name)
        gen_data_command = "futhark dataset -b --u16-bounds="+str(0)+":"+str(size)+" -g "+str_size+"u16 > "+name
        gen_gis_command = "futhark dataset -b -g" + arr_size+"f32 >>" + name
        gen_his_command = "futhark dataset -b -g" + arr_size+"f32 >>" + name
        os.system(gen_data_command)
        os.system(gen_gis_command)
        os.system(gen_his_command)
        fileHandler = open(name, "ab")
        futhark_data.dump(rnd_shape, fileHandler, True)
        futhark_data.dump(size, fileHandler, True)
        #print ("done")
