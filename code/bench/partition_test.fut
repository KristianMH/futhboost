import "../partition"

-- ==
-- entry: test
-- compiled input @ partition_bench_data/test_1000000_20
-- output @ partition_bench_data/result_1000000_20
entry test [n][d][m] (data: [n][d]f32) (shp: [m]i64) (conds: [m]f32) (dims: [m]i64) =
  (partition_lifted (zip dims conds) 0.0 (<) shp data).0
