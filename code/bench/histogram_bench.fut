import "../histogram-based/hist-utils"
import "../util"
-- ==
-- entry: num_ele
-- input @ histogram_bench_data/bench_1000_16
-- input @ histogram_bench_data/bench_10000_32
-- input @ histogram_bench_data/bench_100000_64
-- input @ histogram_bench_data/bench_1000000_128
-- input @ histogram_bench_data/bench_10000000_256
-- input @ histogram_bench_data/bench_20000000_512
entry num_ele [n][d][l] (data: [n][d]u16) (gis: [n]f32) (his: [n]f32)
              (shp: [l]i64) (num_bins: i64) =
  let flag_arr = mkFlagArray shp 0u16 1u16 n
  in
  create_histograms data gis his flag_arr l num_bins


-- ==
-- entry: num_segs
-- input @ histogram_bench_data/bench_16_10000000
-- input @ histogram_bench_data/bench_32_10000000
-- input @ histogram_bench_data/bench_64_10000000
-- input @ histogram_bench_data/bench_256_10000000
-- input @ histogram_bench_data/bench_512_10000000
-- input @ histogram_bench_data/bench_1024_10000000
-- input @ histogram_bench_data/bench_2048_10000000
-- input @ histogram_bench_data/bench_4096_10000000
entry num_segs [n][d][l] (data: [n][d]u16) (gis: [n]f32) (his: [n]f32)
               (shp: [l]i64) (num_bins: i64) =
  let flag_arr = mkFlagArray shp 0u16 1u16 n
  in
  create_histograms data gis his flag_arr l num_bins
