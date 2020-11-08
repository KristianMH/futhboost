import "../histogram-based/bins"

-- input @ bins_bench_data/bench_100

-- ==
-- entry: num_ele
-- input @ bins_bench_data/bench_1000_20
-- input @ bins_bench_data/bench_10000_20
-- input @ bins_bench_data/bench_100000_20
-- input @ bins_bench_data/bench_1000000_20
-- input @ bins_bench_data/bench_2000000_20

entry num_ele [n][d] (data: [n][d]f32) =
  map (\r -> binMap r 256) (transpose data)



-- different bins!!
-- ==
-- entry: num_bins
-- input @ bins_bench_data/bench_1000000_20_16
-- input @ bins_bench_data/bench_1000000_20_32
-- input @ bins_bench_data/bench_1000000_20_64
-- input @ bins_bench_data/bench_1000000_20_128
-- input @ bins_bench_data/bench_1000000_20_256
-- input @ bins_bench_data/bench_1000000_20_512
-- input @ bins_bench_data/bench_1000000_20_1024

entry num_bins [n][d] (data: [n][d]f32) (b: i64) =
  map (\r -> binMap r b) (transpose data)
