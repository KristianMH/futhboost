bins_data:
	python gen_bins_data.py

partition_data:
	python gen_partition_data.py

bins_bench_c: bins_data bins_bench.fut
	futhark bench --backend=c \
                      --json jsons/bins_bench_c.json -r10 \
		  	bins_bench.fut

bins_bench_ele_opencl: bins_data bins_bench.fut
	futhark bench --backend=opencl \
		      --entry=num_ele \
		      --json jsons/bins_bench_ele_opencl.json -r10 \
		      bins_bench.fut

bins_bench_ele_cuda: bins_data bins_bench.fut
	futhark bench --backend=cuda \
		      --entry=num_ele \
		      --json jsons/bins_bench_ele_cuda.json -r10 \
		      bins_bench.fut


partition_bench_ele_cuda: partition_data partition_bench.fut
	futhark bench --backend=cuda \
		      --entry=num_ele \
		      --json jsons/partition_bench_ele_cuda.json -r10 \
		      partition_bench.fut

partition_bench_ele_opencl: partition_data partition_bench.fut
	futhark bench --backend=opencl \
		      --entry=num_ele \
		      --json jsons/partition_bench_ele_opencl.json -r10 \
		      partition_bench.fut

partition_bench_segs_cuda: partition_data partition_bench.fut
	futhark bench --backend=cuda \
		      --entry=num_segs \
		      --json jsons/partition_bench_segs_cuda.json -r10 \
		      partition_bench.fut

partition_bench_segs_opencl: partition_data partition_bench.fut
	futhark bench --backend=cuda \
		      --entry=num_segs \
		      --json jsons/partition_bench_segs_opencl.json -r10 \
		      partition_bench.fut

partition_test: partition_data partition_test.fut
	futhark test partition_test.fut --backend=cuda

# partition_bench_bin_opencl: partition_data partition_bench.fut
# 	futhark bench --backend=opencl \
# 		      --entry=num_partition \
# 		      --json jsons/partition_bench_opencl.json -r10 \
# 		      partition_bench.fut
all:clean

clean:
	find . -maxdepth 1 -type f -executable -exec rm '{}' \;
	rm *.c

clean_tests:
	rm *.actual
	rm *.expected
