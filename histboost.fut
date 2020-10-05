import "lib/github.com/diku-dk/sorts/radix_sort"
import "lib/github.com/diku-dk/segmented/segmented"
import "util"
import "hist"
import "data"
import "tree"
import "partition"
import "bins"
--type dataentry = {}

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

-- return list of (idx, val) (i32, f32) dim id and split val!
-- maybe flag wether it is end? leaf-weight!!! calculation
-- handle missing values ? f32.nan

let train_round_v2 [n][d][b] (data: [d][n]i32) (bin_bounds: [d][b]binboundaries)
                             (labels: [n]f32) (preds: [n]f32) (max_depth: i32) 
                             (l2: f32) (eta: f32) (gamma: f32) = -- : [](i32, f32, bool, bool) =
  let gis = map2 (\p y -> gradient_mse p y) preds labels
  let his = map2 (\p y -> hessian_mse p y) preds labels
  --let data_x = zip3 (replicate n 1) (iota n) data
  let data_x = data
  let active_leafs = [1]
  let shp_x = [n]
  let active_points_idx = iota n
  -- combine active_leafs and shp? most likely
  let (_, res, _, _, _) =  --loop gis and his?
    loop (active_leafs, tree, shp, i, active_idx) =
      (active_leafs, [], shp_x, 0, active_points_idx)
      while (i < max_depth) && !(null active_leafs) do
    -- active gis and his as well? or loop it around
    --let data_active = map (\vs -> permute data active_idx) data
    let l_shp = length shp
    let flag_arr = mkFlagArray shp 0 (replicate l_shp 1) l_shp

    -- let seg_offsets = scanExc (+) 0 flag_arr |> map (*b)
    -- let gis = permute gis active_leafs :> []f32
    -- let new_hists_gis = map2 (\dim_bins -> 
    --                         let idxs = map2 (+) seg_offsets dim_bins
    --                         let hist_entry =  replicate l_shp <| replicate b 0.0f32
    --                         let flat_seg_hist = reduce_by_index (flatten hist_entry)
    --                                                             (+) 0.0 idxs gis
    --                         in unflatten l_shp b
    --                          ) data_active :> [d][l_shp][b]f32
    -- let his = permute his active_leafs
    -- let new_hists_his = map2 (\dim_bins -> 
    --                         let idxs = map2 (+) seg_offsets dim_data
    --                         let hist_entry =  replicate l_shp <| replicate b 0.0f32
    --                         let flat_seg_hist = reduce_by_index (flatten hist_entry)
    --                                                             (+) 0.0 idxs his
    --                         in unflatten l_shp b
    --                          ) data_active :> [d][l_shp][b]f32
    
    in
    (active_leafs, tree, shp ++ [1], i+1, active_idx)
      
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
