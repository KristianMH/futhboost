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
-- input @  partition_bench_data/seg_1000000_16
-- input @  partition_bench_data/seg_1000000_32
-- input @  partition_bench_data/seg_1000000_64
-- input @  partition_bench_data/seg_1000000_128
-- input @  partition_bench_data/seg_1000000_64
-- input @  partition_bench_data/seg_1000000_512
-- input @  partition_bench_data/seg_1000000_1024
-- input @  partition_bench_data/seg_1000000_2048
entry num_segs [n][d][m] (data: [n][d]f32) (shp: [m]i64) (conds: [m]f32) (dims: [m]i64) =
  partition_lifted (zip dims conds) 0.0 (<) shp data
