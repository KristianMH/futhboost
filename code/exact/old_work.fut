import "../lib/github.com/diku-dk/sorts/radix_sort"
import "../util"
import "../tree"
import "../objective"
import "../histogram-based/partition"
import "woop"


let get_leaf_weight [n] (gis: [n]f32) (his: [n]f32) (l2: f32) (eta: f32)
                       : f32 =
  let gsum = reduce (+) 0.0 gis
  let hsum = reduce (+) 0.0 his
  in
  eta*(-gsum/(hsum + l2))-- + min_weight

let gain (gl: f32) (hl:f32) (g:f32) (h:f32) (l2: f32) (gamma: f32)
         (missing_gis_sum: f32) (missing_his_sum: f32): (f32, bool) =
  let gr = g-gl
  let hr = h-hl
  let cost_node = g**2/(h+l2)
  let left_gain_missing = (gl+missing_gis_sum)**2/(missing_his_sum+hl+l2) + gr**2/(hr+l2) - cost_node
  let right_gain_missing = gl**2/(hl+l2)+ (gr+missing_gis_sum)**2/(missing_his_sum+hr+l2) - cost_node
  in
  if left_gain_missing >= right_gain_missing then
    (1/2*left_gain_missing-gamma, true) -- true indicates missing values to left
  else
    (1/2*right_gain_missing-gamma, false) -- falst indicates missing values to right


-- (leaf_idx, data_idx, data)
let do_split [n][d][b] (orig_data: [n](i32, i32, [d]f32)) (li: i32) (leaf_data: [b](i32, i32, [d]f32))
                    (dim: i32) (value: f32) (missing_flag: bool): [n](i32, i32, [d]f32) =
  let (_, data_idx, data) = unzip3 leaf_data
  let new_leaf_idx = map (\data ->
                            if data[i64.i32 dim] < value ||
                               (f32.nan == data[dim] && missing_flag) then
                              li*2
                            else
                              li*2+1) data
  let new_data = zip3 new_leaf_idx data_idx data
  in
  scatter (copy orig_data) (map i64.i32 data_idx) new_data
  --scatter orig_data data_idx new_data



let search_splits2 [l] (data_points: [l]f32) (gis: [l]f32) (his: [l]f32)
                       (l2: f32) (gamma: f32) : (f32, f32, bool) =
  let data = zip3 data_points gis his
  let (missing, rest) = partition (\x -> f32.nan == x.0) data
  --let ha = trace (length rest)
  --let ho = trace (length missing)
  let (_, miss_g, miss_h) = unzip3 missing
  let missing_gis_sum = reduce (+) 0f32 miss_g
  let missing_his_sum = reduce (+) 0f32 miss_h

  let (sorted_data, sorted_gis, sorted_his) =
    radix_sort_float_by_key (.0) f32.num_bits f32.get_bit rest |> unzip3
  -- let unique_seg_starts = map2 (!=) sorted_data (rotate (-1) sorted_data)
  -- let unique_offsets = map i32.bool unique_seg_starts |> scan (+) 0i32 |> map (\t -> t-1)
  -- let num_unique = (last unique_offsets+1)
  -- let dummy_gis = replicate (i64.i32 num_unique) 0f32
  -- let dummy_his = replicate (i64.i32 num_unique) 0f32
  -- let idxs = map i64.i32 unique_offsets
  -- let scan_gis = reduce_by_index dummy_gis (+) 0f32 idxs sorted_gis
  -- let scan_his = reduce_by_index dummy_his (+) 0f32 idxs sorted_his
  -- let sorted_data = zip sorted_data unique_seg_starts |> filter (.1) |> unzip |> (.0)
  let scan_gis = scan (+) 0f32 sorted_gis
  let scan_his = scan (+) 0f32 sorted_his
  let (gm, hm) = (last scan_gis + missing_gis_sum, last scan_his + missing_his_sum)
  let gains = map2 (\g h -> gain g h gm hm l2 gamma missing_gis_sum missing_his_sum)
                   scan_gis scan_his
  let (best_split_idx, max_gain) = (unzip gains).0 |> arg_max
  let xgboost_split_val =
    if best_split_idx < (length sorted_data)-1 then
    --let ha = sorted_data[best_split_idx-10:best_split_idx+10]
    --let (he, _) = unzip gains
    --let ha = trace (zip ha he[best_split_idx-10:best_split_idx+10])
    --in
      (sorted_data[best_split_idx] + sorted_data[best_split_idx+1])/2
    else
      sorted_data[best_split_idx]*2 -- apprently multiply with 2 in xgboost impl
  in 
    (max_gain, xgboost_split_val, gains[best_split_idx].1)







let train_round [n][d] (data: [n][d]f32) (labels: [n]f32) (preds: [n]f32) (max_depth: i64) 
                       (l2: f32) (eta: f32) (gamma: f32) : [](i64, f32, bool, bool) =
  let gis = map2 (\p y -> gradient_mse p y) preds labels
  let his = map2 (\p y -> hessian_mse p y) preds labels
  --let p_idxs = iota n |> map i32.i64
  --let data_x = zip3 (replicate n 1i32) p_idxs data
  let leafs = [1i32]
  let tree = mktree max_depth (0i64, f32.nan, false, false)
  let (_, _, _, final_tree, _, _, _) =
    loop (active_leafs, shp, i, tree, data, gis, his) =
      (leafs, [n], 0, tree, data, gis, his) while !(null active_leafs) && i <= max_depth do
        --let (new_leafs, new_shp, new_tree, new_data, new_gis, new_his) =
        -- fix possible compiler warnings
        let active_points_length = length gis
        let gis  = (gis  :> [active_points_length]f32)
    	let his  = (his  :> [active_points_length]f32)
        let data = (data :> [active_points_length][d]f32)

        let num_leafs = length active_leafs
        let active_leafs = active_leafs :> [num_leafs]i32
        let shp = shp :> [num_leafs]i64
        let data_offsets = scanExc (+) 0i64 shp
        let temp_conds = replicate num_leafs (-1i64, 0f32)
        --new_shp,
        -- conds (dim, t) ... negative dim means done segment/leaf i.e. no split?
        let (new_tree, new_conds) =  
          loop (new_tree, conds) = (tree, temp_conds) for li < num_leafs do
            --, new_data, new_gis, new_his) = 
            --([], [], tree, data, gis, his) for li < num_leafs do
            
            let leaf_idx = active_leafs[li]
            let offset = data_offsets[li]
            let num_points = shp[li]
            let data_in_leaf = data[offset:offset+num_points]
            let gis = gis[offset:offset+num_points]
            let his = his[offset:offset+num_points]
            in
              if i == max_depth then
                let weight = get_leaf_weight gis his l2 eta
                let node = [(0i64, weight, false, false)]
                let updated_tree = scatter new_tree [i64.i32 leaf_idx-1] node
                in
                (updated_tree, conds)
              else
                let pos_splits =
                  map (\feature_vals ->
                         search_splits2 feature_vals gis his l2 gamma
                      ) (transpose data_in_leaf) :> [d](f32, f32, bool)
                --let ha = trace pos_splits
                let (best_split_idx, best_gain) = (unzip3 pos_splits).0 |> arg_max
                let (node, n_conds) = 
                  if best_gain > 0f32 then
                    let split_val = pos_splits[best_split_idx].1
                    let missing_dir = pos_splits[best_split_idx].2
                    let node = [(best_split_idx, split_val, true, missing_dir)]
                    let n_conds = scatter conds [li] [(best_split_idx, split_val)]
                    in
                      (node, n_conds)
                  else
                    --(save parent info for splits?? can save this calculation)
                    let weight = get_leaf_weight gis his eta l2
                    in ([(0i64, weight, false, false)], conds)
                let updated_tree = scatter new_tree [i64.i32 leaf_idx-1] node
                in
                (updated_tree, n_conds)
        let new_conds = new_conds :> [num_leafs](i64, f32)
        let active_leaf_flags = map (\x -> x.0 >= 0) new_conds
                                    --if statement on level?
        let (active_leafs, active_shp, active_conds) =
          zip3 active_leafs shp new_conds |> filter (\x -> (x.2).0 >= 0) |> unzip3
        -- soft max of d at u16 i.e. 65536
        let flag_arr = mkFlagArray shp 0u16 1u16 active_points_length
        let seg_offsets = scan (+) 0u16 flag_arr |> map (\x -> x-1)
        let active_flags = map (\i -> active_leaf_flags[i64.u16 i]) seg_offsets
        let (active_data, active_gis, active_his, _ ) =
          zip4 data gis his active_flags |> filter (.3) |> unzip4
        let (idxs, split_shape, _) = partition_lifted_idx active_conds 0f32 (<) active_shp
                                                    active_data
        let num_active = length active_data
        let new_data = scatter2D (replicate num_active (replicate d 0f32))
                                 idxs active_data
        let temp = zip active_gis active_his
        let (new_gis, new_his) = scatter (replicate num_active (0f32, 0f32)) idxs temp |> unzip
        let new_shp = calc_new_shape active_shp split_shape
        let he = trace new_shp
        let new_leafs = map getChildren active_leafs |> flatten
        let he = trace new_leafs
        let ha = trace active_conds
        --let ha = trace (reduce (+) 0f32 (map (\x -> x[8]) new_data[new_shp[0]:]))
        in
          (new_leafs, new_shp, i+1, new_tree, new_data, new_gis, new_his)
  in
    final_tree


        
let error (label: f32) (pred: f32) : f32 = (label-pred)**2

let train [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i64) (n_rounds: i64)
                       (l2: f32) (eta: f32) (gamma: f32) : [n_rounds]f32 =
  let inital_preds = replicate n 0.5
  let res1 = replicate n_rounds 0.0
  -- scatter to save all trees and return history over n rounds! [n_rounds][2**d](i32,f32,bool)
  -- pad tree before scatter... easiest then simple scatter!
  let res =
    loop (data, labels, preds, e) = (data, labels, inital_preds, res1) for i < n_rounds do
      let tree  = train_round data labels preds max_depth l2 eta gamma |> trace
      --:[](i32, f32, bool, bool)
      let new_preds = map (\x -> predict x tree) data |> map2 (+) preds --|> trace
      let train_error = reduce (+) 0.0 <| map2 (\l p -> error l p) labels new_preds
      let train_error = f32.sqrt (train_error/ (f32.i64 n))
      --let ha = trace train_error
      let res1 = scatter e [i] [train_error]

      in
      (data, labels, new_preds, res1)
  in
  res.3
          
--let eval = train data[:,:2] data[:,2] 3 3 0.5 0.3 0
             
let main [n][d] (data: [n][d]f32) (labels: [n]f32) = train data labels 3 100 0.5 0.3 0
     
let test = train woopdata wooptarget 1 1 0.5 0.3 0
