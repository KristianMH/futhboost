import "partition"
import "bins"



let train_round_v2 [n][d][b] (data: [d][n]i32) (bin_bounds: [d][b]binboundaries)
                             (labels: [n]f32) (preds: [n]f32) (max_depth: i32) 
                             (l2: f32) (eta: f32) (gamma: f32) = -- : [](i32, f32, bool, bool) =
  let data_x = data
  let active_leafs = [1]
  let shp_x = [n]
  let active_points_idx = iota n
  -- combine active_leafs and shp? most likely
  let (res, _, _, _) =  --loop gis and his?
    loop (active_leafs, tree, shp, i, active_idx) =
      (active_leafs, [], shp_x, 0, active_points_idx)
      while (i < max_depth) && !(null active_leafs) do
    let l_shp = length shp
    let flag_arr = mkFlagArray shp 0 1 l_shp
    in
    (active_leafs, tree, shp ++ [1], i+1, active_idx)
      
  in
  res
