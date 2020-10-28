bins_data:
	python gen_bins_data.py

partition_data:
	python gen_partition_data.py

bins_bench_c: bins_data bins_bench.fut
	futhark bench --backend=c \
                      --json bins_bench_c.json -r10 \
		  	bins_bench.fut

bins_bench_ele_opencl: bins_data bins_bench.fut
	futhark bench --backend=opencl \
		      --entry=num_ele \
		      --json bins_bench_opencl.json -r10 \
		      bins_bench.fut

bins_bench_ele_cuda: bins_data bins_bench.fut
	futhark bench --backend=cuda \
		      --entry=num_ele \
		      --json bins_bench_cuda.json -r10 \
		      bins_bench.fut


partition_bench_ele_cuda: partition_data partition_bench.fut
	futhark bench --backend=cuda \
		      --entry=num_ele \
		      --json partition_bench_cuda.json -r10 \
		      partition_bench.fut

partition_bench_ele_opencl: partition_data partition_bench.fut
	futhark bench --backend=opencl \
		      --entry=num_ele \
		      --json partition_bench_opencl.json -r10 \
		      partition_bench.fut

# partition_bench_bin_opencl: partition_data partition_bench.fut
# 	futhark bench --backend=opencl \
# 		      --entry=num_partition \
# 		      --json partition_bench_opencl.json -r10 \
# 		      partition_bench.fut
