import "lib/github.com/diku-dk/sorts/radix_sort"
import "util"
import "data"
import "tree"
--type dataentry = {}

let gradient_mse (pred: f32) (orig: f32): f32 = 1*(pred-orig)

let hessian_mse 't (pred: t) (orig: t): f32 =  1.0

let cost (gis: []f32) (his: []f32) (lamda: f32): f32 =
  let gsum = reduce (+) 0.0 gis
  let hsum = reduce (+) 0.0 his
  in
  - gsum**2 / (2.0*(hsum + lamda))

-- add eta?
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
                            if data[dim] < value || (f32.nan == data[dim] && missing_flag) then
                              li*2
                            else
                              li*2+1) data
  let new_data = zip3 new_leaf_idx data_idx data
  in
  -- why copy??!!!
  scatter (copy orig_data) data_idx new_data
  --scatter orig_data data_idx new_data

-- returns split value and gain
let search_splits [n] [l] [d] (data_points: [l][d]f32) (point_idx: [l]i32)
                          (gis: [n]f32) (his: [n]f32) (dim: i32) (l2: f32) (gamma: f32)
                          : (f32, f32, bool) = 
  let data_dim = data_points[:,dim]
  let (missing, rest_data) = partition (\x -> f32.nan ==x.1) (zip point_idx data_dim)
  let missing_gis_sum = reduce (+) 0.0 <| map (\x -> gis[x.0]) missing
  let missing_his_sum = reduce (+) 0.0 <| map (\x -> his[x.0]) missing

  let sorted = radix_sort_float_by_key (.1) f32.num_bits f32.get_bit
                                       rest_data
  let (sorted_idx, sorted_data) = unzip sorted
  let sorted_gis = map (\i -> gis[i]) sorted_idx
  let sorted_his = map (\i -> his[i]) sorted_idx
  let scan_gis = scan (+) 0.0 sorted_gis
  let scan_his = scan (+) 0.0 sorted_his
  let (gm, hm) = (last scan_gis + missing_gis_sum, last scan_his + missing_his_sum)
  let gains = map2 (\g h -> gain g h gm hm l2 gamma missing_gis_sum missing_his_sum)
                   scan_gis scan_his
  let (best_split_idx, max_gain) = (unzip gains).0 |> arg_max
  --let ha = trace best_split_idx
  -- check bounds?
  let xgboost_split_val = if best_split_idx < (length sorted_idx)-1 then
                            (sorted_data[best_split_idx] + sorted_data[best_split_idx+1])/2
                          else
                            sorted_data[best_split_idx]*2 -- apprently multiply with 2 in xgboost impl
  --let ha = trace (xgboost_split_val, max_gain, best_split_idx < (length sorted_idx)-1)
  in
    --splits ++ [(sorted_data[best_split_idx], max_gain)]
  --splits ++ [(xgboost_split_val, max_gain)]
  (xgboost_split_val, max_gain, gains[best_split_idx].1)



let getChildren (i: i32): [2]i32 =
  [2*i, 2*i+1]
-- return list of (idx, val) (i32, f32) dim id and split val!
-- maybe flag wether it is end? leaf-weight!!! calculation
-- handle missing values ? f32.nan
let train_round [n][d] (data: [n][d]f32) (labels: [n]f32) (preds: [n]f32) (max_depth: i32) 
                       (l2: f32) (eta: f32) (gamma: f32) : [](i32, f32, bool, bool) =
  let gis = map2 (\p y -> gradient_mse p y) preds labels
  let his = map2 (\p y -> hessian_mse p y) preds labels
  let data_x = zip3 (replicate n 1) (iota n) data
  let active_leafs = [1]
  let (_, res, _, _) =
    loop (active_leafs, tree, data, leaf_count) = (active_leafs,[], data_x, 0)
    while !(null active_leafs) && (log2 (leaf_count) <= max_depth) do
      let leaf_idx = head active_leafs
      let points_in_leaf = filter (\x -> x.0 == leaf_idx) data
      in
      if length points_in_leaf == 1 then -- cannot split node with one ele :) -- min size?
        let point_idx = (head points_in_leaf).1
        let weight = eta*(-gis[point_idx]/(his[point_idx]+l2)) --+ min_weight
        in
        (tail active_leafs, tree ++ [(0, weight, false, false)], data, leaf_count+2)
      else
        let (_, point_idx, data_points) = unzip3 points_in_leaf
        --let ha = trace data_points
        let pos_splits = map (\i -> search_splits data_points point_idx gis his i l2 gamma) (iota d)
        let (vals, gains, missing_flags) = unzip3 pos_splits
        --let gains = if leaf_count > 1 then gains[1:] else gains
        let (dim, gain) = arg_max gains
        let value = vals[dim]
        let missing_flag = missing_flags[dim]
        let leaf_count = leaf_count+2
        let node_flag = (gain > 0.0) && (log2 (leaf_count) < max_depth)
        let (value, data, new_leafs) =
          if node_flag then
          let new_data = do_split data leaf_idx points_in_leaf dim value missing_flag
          let new_leafs = (tail active_leafs)  ++  (getChildren leaf_idx)
          in
          (value, new_data, new_leafs)
          else
          let gis = map (\i -> gis[i]) point_idx
          let his = map (\i -> his[i]) point_idx
          let weight = get_leaf_weight gis his l2 eta
          in
          (weight, data, tail active_leafs)
        in
        (new_leafs, tree++[(dim, value, node_flag, missing_flag)], data, leaf_count)
  in
  res


-- return list of (idx, val) (i32, f32) dim id and split val!
-- maybe flag wether it is end? leaf-weight!!! calculation
-- handle missing values ? f32.nan
let train_round_v2 [n][d] (data: [n][d]f32) (labels: [n]f32) (preds: [n]f32) (max_depth: i32) 
                       (l2: f32) (eta: f32) (gamma: f32) : [](i32, f32, bool, bool) =
  let gis = map2 (\p y -> gradient_mse p y) preds labels
  let his = map2 (\p y -> hessian_mse p y) preds labels
  let data_x = zip3 (replicate n 1) (iota n) data
  let active_leafs = [1]
  let (_, res, _) =
    loop (active_leafs, tree, data) = (active_leafs, [], data_x) for l < max_depth do
      let (leaf_min, leaf_max) = if l == 0 then (1,1) else (2**l, 2**(l+1)-1)
      let leafs_to_process= filter (\i-> leaf_min <= i && i<= leaf_max) active_leafs
      let ha = trace (leafs_to_process)
      let (current_leafs, new_tree, data, new_leafs) = 
        loop (active_leafs, tree, data, new_leafs) = (leafs_to_process, [], data, [])
        while !(null active_leafs) do
          let leaf_idx = trace (head active_leafs)
          let points_in_leaf = filter (\x -> x.0 == leaf_idx) data
          in
            if length points_in_leaf == 1 then -- cannot split node with one ele :) -- min size?
              let point_idx = (head points_in_leaf).1
              let weight = eta*(-gis[point_idx]/(his[point_idx]+l2)) --+ min_weight
              in
              (tail active_leafs, tree ++ [(0, weight, false, false)], data, new_leafs)
            else
              let (_, point_idx, data_points) = unzip3 points_in_leaf
              --let ha = trace data_points
              let pos_splits = map (\i -> search_splits data_points point_idx gis
                                                        his i l2 gamma) (iota d)
              let (vals, gains, missing_flags) = unzip3 pos_splits
              let (dim, gain) = arg_max gains
              let value = vals[dim]
              let missing_flag = missing_flags[dim]
              let node_flag = (gain > 0.0)
              let (value, data, new_leafs) =
                if node_flag then
                  let new_data = do_split data leaf_idx points_in_leaf dim value missing_flag
                  let new_leafs = new_leafs  ++  (getChildren leaf_idx)
                  in
                  (value, new_data, new_leafs)
                else
                  let gis = map (\i -> gis[i]) point_idx
                  let his = map (\i -> his[i]) point_idx
                  let weight = get_leaf_weight gis his l2 eta
                  in
                  (weight, data, if null new_leafs then new_leafs else tail new_leafs)
              in
              (trace (tail active_leafs), tree++[(dim, value, node_flag, missing_flag)], data, new_leafs)
      in
      (new_leafs, tree ++ new_tree, data)
      
  in
  res
let error (label: f32) (pred: f32) : f32 = (label-pred)**2

let train [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i32) (n_rounds: i32)
                       (l2: f32) (eta: f32) (gamma: f32) : f32 =
  let inital_preds = replicate n 0.5
  -- scatter to save all trees and return history over n rounds! [n_rounds][2**d](i32,f32,bool)
  -- pad tree before scatter... easiest then simple scatter!
  let res =
    loop (data, labels, preds, e) = (data, labels, inital_preds, 0) for i < n_rounds do
      let tree  = train_round_v2 data labels preds max_depth l2 eta gamma |> trace
      --:[](i32, f32, bool, bool)
      let new_preds = map (\x -> predict x tree) data |> map2 (+) preds |> trace
      let train_error = reduce (+) 0.0 <| map2 (\l p -> error l p) labels new_preds
      let train_error = f32.sqrt (train_error/ (f32.i32 n))
      let ha = trace train_error
      in
      (data, labels, new_preds, train_error)
  in
  res.3
          
let eval = train data[:,:2] data[:,2] 3 3 0.5 0.3 0
             
--let main (xs: [][]f32) = let res = train xs[:,:2] xs[:,2] in res[0].1
