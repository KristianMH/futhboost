import "../partition"

-- input @ bins_bench_data/bench_100
-- input @ partition_bench_data/bench_100000000_20
-- tags {disable}

                
-- ==
-- entry: num_ele
-- input @ partition_bench_data/bench_1000_20
-- input @ partition_bench_data/bench_10000_20
-- input @ partition_bench_data/bench_100000_20
-- input @ partition_bench_data/bench_1000000_20
-- input @ partition_bench_data/bench_10000000_20


entry num_ele [n][d][m] (data: [n][d]f32) (shp: [m]i64) (conds: [m]f32) (dims: [m]i64) =
  partition_lifted (zip dims conds) 0.0 (<) shp data

-- ==
-- entry: num_segs
-- input @  partition_bench_data/seg_16_1000000
-- input @  partition_bench_data/seg_32_1000000
-- input @  partition_bench_data/seg_64_1000000
-- input @  partition_bench_data/seg_128_1000000
-- input @  partition_bench_data/seg_64_1000000
-- input @  partition_bench_data/seg_512_1000000
-- input @  partition_bench_data/seg_1024_1000000
-- input @  partition_bench_data/seg_2048_1000000
entry num_segs [n][d][m] (data: [n][d]f32) (shp: [m]i64) (conds: [m]f32) (dims: [m]i64) =
  partition_lifted (zip dims conds) 0.0 (<) shp data
