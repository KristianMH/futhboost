import json
import sys
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

_, seq, par = sys.argv

seq_json = json.load(open(seq), object_pairs_hook=OrderedDict)
par_json = json.load(open(par), object_pairs_hook=OrderedDict)

speedups = {}

def dataset_to_n(dataset):
    return int(dataset.split("_")[-2])


sizes = []
ms_seq = []
ms_par = []

for ((prog, run), (prog1, run1)) in zip(seq_json.items(), par_json.items()):
    seq_datasets = run["datasets"]
    par_datasets = run1["datasets"]
    for ((dataset, seq_run_res), (dataset1, par_run_res)) in zip(seq_datasets.items(), par_datasets.items()):
        #print("processing dataset:", dataset, dataset1)
        #print(seq_run_res)
        try: # handle odd opencl error
            seq_run_times = seq_run_res["runtimes"]
            par_run_times = par_run_res["runtimes"]
        except: 
            continue
        #seq_run_times = seq_run_res["runtimes"]
        
        seq_run_avg = np.mean(seq_run_times)
        par_run_avg = np.mean(par_run_times)
        num_eles = dataset_to_n(dataset)
        num_eles1 = dataset_to_n(dataset1)
        if num_eles != num_eles1:
            print("OHHH FUCKKK!", num_eles, num_eles1)
        sizes.append(num_eles)
        ms_seq.append(seq_run_avg/1000)
        ms_par.append(par_run_avg/1000)
        #print("speedup:", seq_run_avg/par_run_avg, num_eles)

sizes = np.array(sizes)
ms_seq = np.array(ms_seq)
ms_par = np.array(ms_par)
sortedidxs = np.argsort(sizes)
sizes = sizes[sortedidxs]
ms_seq = ms_seq[sortedidxs]
ms_par = ms_par[sortedidxs]
#print(ms_par)
#print(sizes)
fig, ax = plt.subplots()
ax.plot(sizes, ms_par, color="blue", label="openCL")
ax.plot(sizes, ms_seq, color="red", label="cuda")
ax.set_xscale("log", base=10)#2)
ax.set_xlabel("Size of input")
ax.set_ylabel("Runtime(ms)")
ax.set_title(seq +" vs "+ par)

ax.legend()
plt.savefig("partition_bench_segs.png")
