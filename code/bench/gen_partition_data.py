import numpy as np
import futhark_data
import os

path = "partition_bench_data/"
try:
    os.mkdir(path)
except:
    print("dir already exists")
    pass
prefix = "bench"
NS = [10**3, 10**4, 10**5, 10**6, 10**7, 10**8]



def path_name(path, prefix, n,k):
    return path+prefix+"_"+str(n)+"_"+str(k)

def file_name(prefix, n, k):
    return prefix+"_"+str(n)+"_"+str(k)
def matsize_to_str(n,k):
    ks = "["+str(k)+"]"
    ns = "["+str(n)+"]"
    return ns + ks

datasets = os.listdir(path)

size = 20
n = 128 #number of segments. try vary
for num in NS:
    str_size = matsize_to_str(num, size)
    name = path_name(path, prefix, num, size)
    # #print(file_name(prefix, num, size) not in datasets)
    if file_name(prefix, num, size) not in datasets:
        #data = np.random.rand(num,size).astype("float32")
        #print(data.shape, data.dtype)
        rnd_shape = np.random.multinomial(num, np.ones(n)/n, size=1)[0].astype("int64")
        #print(rnd_shape.shape, rnd_shape.dtype)
        conds = np.random.rand(n).astype("float32")
        #print(conds.shape, conds.dtype)
        split_idxs = np.random.randint(size-1, size=n).astype("int64")
        #print(split_idxs.shape, split_idxs.dtype)
        #print("making "+str_size+" matrix with name "+ name)
        print("Making: "+name)
        os.system("futhark dataset -b -g "+str_size+"f32 > "+name)
        fileHandler = open(name, "ab")
        #futhark_data.dump(data, fileHandler, True)
        futhark_data.dump(rnd_shape, fileHandler, True)
        futhark_data.dump(conds, fileHandler, True)
        futhark_data.dump(split_idxs, fileHandler, True)
    #     os.system("futhark dataset -b -g "+str_size+"f32 > "+name)
    #     print ("done")

SEGS = [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11]    
size = 20
num = 10**6 #number of elements
prefix = "seg"
for seg in SEGS:
    str_size = matsize_to_str(num, size)
    name = path_name(path, prefix, seg, num)
    # #print(file_name(prefix, num, size) not in datasets)
    if file_name(prefix, seg, num) not in datasets:
        #print("writing: "+name)
        #data = np.random.rand(num,size).astype("float32")
        #print(data.shape, data.dtype)
        rnd_shape = np.random.multinomial(num, np.ones(seg)/seg, size=1)[0].astype("int64")
        #print(rnd_shape.shape, rnd_shape.dtype)
        conds = np.random.rand(seg).astype("float32")
        #print(conds.shape, conds.dtype)
        split_idxs = np.random.randint(size-1, size=seg).astype("int64")
        #print(split_idxs.shape, split_idxs.dtype)
        #print("making "+str_size+" matrix with name "+ name)
        print("Making:" + name)
        os.system("futhark dataset -b -g "+str_size+"f32 > "+name)
        fileHandler = open(name, "ab")
        #futhark_data.dump(data, fileHandler, True)
        futhark_data.dump(rnd_shape, fileHandler, True)
        futhark_data.dump(conds, fileHandler, True)
        futhark_data.dump(split_idxs, fileHandler, True)
        #     os.system("futhark dataset -b -g "+str_size+"f32 > "+name)
        #     print ("done")
    
def split(arr, cond):
    #return [arr[cond], arr[~cond]]#.flatten()
    return np.concatenate((arr[cond], arr[~cond]), axis=0)

def partition(arr, shp, conds, split_idxs):
    rot = np.roll(shp, 1)
    rot[0] = 0
    summed_shp = np.cumsum(rot)
    #print(summed_shp,shp)
    #res = np.array([[0,0]])
    res = []
    for (c,s, val, dim) in zip(summed_shp, shp, conds, split_idxs):
        dat = arr[c:c+s]
        #for entry in dat:
        temp = split(dat, dat[:, dim] < val)
        res.append(temp)
    return np.vstack(res)

# a = np.array([[1,2,3],[4,5,6],[7,8,9],[2,4,7]])
# print(split(a, a[:,0]<3))
# d = np.array([[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]])
# s = [2,3]
# c = [3,4]
# l = [1,0]
# print(partition(d, s, c, l)) 

        
n = 10**6
size = 20
num_segs=50
str_size = matsize_to_str(n, size)
name = path_name(path, "test", n, size)
if file_name("test", n, size) not in datasets:
    data = np.random.rand(n,size).astype("float32")
    #print(data.shape, data.dtype)
    rnd_shape = np.random.multinomial(n, np.ones(num_segs)/num_segs, size=1)[0].astype("int64")
    #print(rnd_shape.shape, rnd_shape.dtype)
    conds = np.random.rand(num_segs).astype("float32")
    #print(conds.shape, conds.dtype)
    split_idxs = np.random.randint(size-1, size=num_segs).astype("int64")
    print("writing test data")
    fileHandler = open(name, "wb")
    futhark_data.dump(data, fileHandler, True)
    futhark_data.dump(rnd_shape, fileHandler, True)
    futhark_data.dump(conds, fileHandler, True)
    futhark_data.dump(split_idxs, fileHandler, True)
    print("wrote test data")
    res = partition(data, rnd_shape, conds, split_idxs)
    resultHandler = open(path_name(path, "result", n, size), "wb")
    futhark_data.dump(res, resultHandler, True)
    print("wrote result data")
