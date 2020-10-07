import "lib/github.com/diku-dk/sorts/radix_sort"
import "lib/github.com/diku-dk/segmented/segmented"
import "util"
import "hist"
import "data"
import "tree"
import "partition"
import "bins"

type node_vals = (f32, f32) -- gsum, hsum

let gradient_mse (pred: f32) (orig: f32): f32 = 1*(pred-orig)

let hessian_mse 't (pred: t) (orig: t): f32 =  1.0

let cost (gis: []f32) (his: []f32) (lamda: f32): f32 =
  let gsum = reduce (+) 0.0 gis
  let hsum = reduce (+) 0.0 his
  in
  - gsum**2 / (2.0*(hsum + lamda))

let get_leaf_weight [n] (gis: [n]f32) (his: [n]f32) (l2: f32) (eta: f32)
                       : f32 =
  let gsum = reduce (+) 0.0 gis
  let hsum = reduce (+) 0.0 his
  in
  eta*(-gsum/(hsum + l2))-- + min_weight


let gain (gl: f32) (hl: f32) (g: f32) (h: f32) (l2: f32) (gamma: f32) : (f32, bool) =
  let gr = g-gl
  let hr = h-hl
  in (1/2*(gl**2/(hl+l2)+gr**2/(hr+l2)-g**2/(h+l2)), true)

let getChildren (i: i32): [2]i32 =
  [2*i, 2*i+1]



-- find the best split within a dimension
-- returns (gain, split_val, missing_dir, node_left, node_right)
let find_split_hist [m] (g_hist: [m]f32) (h_hist: []f32) (bin_bounds: [m]binboundaries)
                        (g: f32) (h: f32) (l2: f32) (gamma: f32)
                        : (f32, f32, bool, node_vals, node_vals) =
  let gls = scan (+) 0.0 g_hist
  let hls = scan (+) 0.0 h_hist
  let gains = map2 (\gl hl -> gain gl hl g h l2 gamma) gls hls
  let (gains, flags) = unzip gains
  let (best_split_idx, best_gain) = arg_max gains
  let split_val = bin_bounds[best_split_idx] |> (.1) -- max.
  let node_left = (gls[best_split_idx], hls[best_split_idx])
  let node_right = tuple_math (-) (g,h) node_left
  in (best_gain, split_val, flags[best_split_idx], node_left, node_right)


-- reduce with map2 should be implemented I think- work in progress
let find_best_splits [d][s] (splits: [d][s](f32, f32, bool, node_vals, node_vals))
                            : [s](f32, f32, bool, node_vals, node_vals) =
  splits[0]

-- maps over each dim -> map over each segment, everything should be regular with histograms
-- returns (dim_idx, split_val, is_leaf?, missing_dir, node_left, node_right)
let search_splits_segs [d][s][m] (g_hists: [d][s][m]f32) (h_hists: [d][s][m]f32)
                              (g_node: [s]f32) (h_node: [s]f32)
                              (bin_bounds: [d][m]binboundaries) (l2: f32) (gamma: f32)
                              : [s](i32, f32, bool, bool, node_vals, node_vals) =
  let best_splits_dim =
    map3 (\seg_g_hist seg_h_hist bin_bound -> -- map over each dim
            map4 (\g_hist h_hist g h -> -- map over each segment
                    find_split_hist g_hist h_hist bin_bound g h l2 gamma)
                 seg_g_hist seg_h_hist g_node h_node 
         ) g_hists h_hists bin_bounds :> [d][s](f32, f32, bool, node_vals, node_vals)
  let best_splits = find_best_splits best_splits_dim
  -- need to add terminal leaf flag but then done.
  in
  replicate s (0, 0.0, false, false, (0.0,0.0), (0.0,0.0))





-- return list of (idx, val) (i32, f32) dim id and split val!
-- maybe flag wether it is end? leaf-weight!!! calculation
-- handle missing values ? f32.nan


-- data layout. [d][n] great for hist calculation but not for partition_lifted
-- perfer [n][d]
let train_round [n][d][b] (data: [d][n]i32) (bin_bounds: [d][b]binboundaries)
                             (labels: [n]f32) (preds: [n]f32) (max_depth: i32) 
                             (l2: f32) (eta: f32) (gamma: f32) = -- : [](i32, f32, bool, bool) =
  let tree = replicate (2**max_depth-1) (0,f32.nan, false, false) :> [](i32, f32, bool, bool)
  let gis = map2 (\p y -> gradient_mse p y) preds labels
  let his = map2 (\p y -> hessian_mse p y) preds labels
  --let data_x = zip3 (replicate n 1) (iota n) data
  let data_x = data
  let active_points_idx = iota n
  -- leaf consist of id, #num_elements and (G, H) sums
  let root = zip3 [1] [n] [(0, 0)]
  -- combine active_leafs and shp? most likely
  let (_, res, _, _, _, _, _ ) =  --loop gis and his?
    loop (leafs, tree, i, data, gis, his, active_points_idx) =
      (root, tree, 0, data_x, gis, his, active_points_idx)
      while (i < max_depth) && !(null leafs) do
    -- active gis and his as well? or loop it around
    --let data_active = map (\vs -> permute data active_idx) data
    let (shp, leaf_id_active, GH) = unzip3 leafs -- leaf data, shp is num points in each seg
    let l_shp = length shp -- #num leafs
    let (GS, HS) = unzip GH -- parent infomation for each segment
    let flag_arr = mkFlagArray shp 0 1 (length active_points_idx)
    -- seg_offsets are multiplied with #num_bins to fit flat representation of [#segs][b] hists
    -- flatten unflatten are used or not?
    let seg_offsets = scanExc (+) 0 flag_arr |> map (*b)
    let new_hists_gis = map (\dim_bins -> 
                            let idxs = map2 (+) seg_offsets dim_bins
                            let hist_entry =  replicate (l_shp*b) 0.0f32
                            let flat_seg_hist = reduce_by_index hist_entry
                                                                (+) 0.0 idxs gis
                            in unflatten l_shp b flat_seg_hist
                             ) data --:> [d][l_shp][b]f32
    let new_hists_his = map (\dim_bins -> 
                            let idxs = map2 (+) seg_offsets dim_bins
                            let hist_entry =  replicate (l_shp*b) 0.0f32
                            let flat_seg_hist = reduce_by_index hist_entry
                                                                (+) 0.0 idxs his
                            in unflatten l_shp b flat_seg_hist
                             ) data --:> [d][l_shp][b]f32

    let splits = search_splits_segs new_hists_gis new_hists_his GS HS bin_bounds
                                    l2 gamma
    -- splits should be [s](i32, f32, bool, bool, node_vals, node_vals)
                      -- (filter out terminal leafs)
    -- then partition_lifted can be used with [s](i32, f32) on data (i.e. do_splits)
    -- new leafs can be calulated from partiton
    -- should data, gis, his be updated or keep an active_points array
    -- and then load everything at start of loop?

    -- tree will be updated with scatter
    in
    (leafs, tree, i+1, data, gis, his, active_points_idx)
      
  in
  res



let error (label: f32) (pred: f32) : f32 = (label-pred)**2

-- let train [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i32) (n_rounds: i32)
--                        (l2: f32) (eta: f32) (gamma: f32) : f32 =
--   let inital_preds = replicate n 0.5
--   let res =
--     loop (data, labels, preds, e) = (data, labels, inital_preds, 0) for i < n_rounds do
--       let tree  = train_round_v2 data labels preds max_depth l2 eta gamma
--       --:[](i32, f32, bool, bool)
--       let new_preds = map (\x -> predict x tree) data |> map2 (+) preds 
--       let train_error = reduce (+) 0.0 <| map2 (\l p -> error l p) labels new_preds
--       let train_error = f32.sqrt (train_error/ (f32.i32 n))
--       --let ha = trace train_error
--       in
--       (data, labels, new_preds, train_error)
--   in
--   res.3
          
-- let eval = train data[:,:2] data[:,2] 3 3 0.5 0.3 0
             
--let main (xs: [][]f32) = let res = train xs[:,:2] xs[:,2] in res[0].1
