import os

NS = [10**3, 10**4, 10**5, 10**6, 10**6*2]


path = "bins_bench_data/"
try:
    os.mkdir(path)
except:
    print("dir already exists")
    pass
prefix = "bench"
def matsize_to_str(n,k):
    ks = "["+str(k)+"]"
    ns = "["+str(n)+"]"
    return ns + ks


def path_name(path, prefix, n,k):
    return path+prefix+"_"+str(n)+"_"+str(k)

def file_name(prefix, n, k):
    return prefix+"_"+str(n)+"_"+str(k)


datasets = os.listdir(path)
#print( datasets)
size = 20
for num in NS:
    #for size in MAT_SIZES:
    str_size = matsize_to_str(num, size)
    name = path_name(path, prefix, num, size)
    #print(file_name(prefix, num, size) not in datasets)
    if file_name(prefix, num, size) not in datasets:
        print("making "+str_size+" matrix with name "+ name)
        os.system("futhark dataset -b -g "+str_size+"f32 > "+name)
        print ("done")

        
# num_bins = [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10]
# num = 10**6
# for b in num_bins:
#     b = str(b)
#     #for size in MAT_SIZES:
#     str_size = matsize_to_str(num, size)
#     name = path_name(path, prefix, num, size)
#     #print(file_name(prefix, num, size) not in datasets)
#     if file_name(prefix, num, size) not in datasets:
#         fn = name+"_"+b
#         bound = "--i64-bounds="+b+":"+b+" -g i64"
#         print("making "+str_size+" matrix with name "+ fn)
#         os.system("futhark dataset -b -g "+str_size+"f32 "+bound+" > "+fn)
#         print ("done")
