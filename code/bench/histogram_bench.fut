import "../histogram-based/hist-utils"
import "../util"
-- ==
-- entry: num_ele
-- input @ histogram_bench_data/bench_1000_16

entry num_ele [n][d][l] (data: [n][d]u16) (gis: [n]f32) (his: [n]f32)
              (shp: [l]i64) (num_bins: i64) =
  let flag_arr = mkFlagArray shp 0i64 1i64 n
  in
  create_histograms data gis his flag_arr l num_bins
