import "lib/github.com/diku-dk/sorts/radix_sort"
import "util"
import "data"
import "tree"
import "objective"
--type dataentry = {}


let getChildren (i: i32): [2]i32 = [2*i, 2*i+1]
-- add eta?
let get_leaf_weight [n] (gis: [n]f32) (his: [n]f32) (l2: f32) (eta: f32)
                       : f32 =
  let gsum = reduce (+) 0.0 gis
  let hsum = reduce (+) 0.0 his
  in
  eta*(-gsum/(hsum + l2))-- + min_weight3


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

-- search_splits data_points[:,i] point_idxs gis his l2 gamma
let search_splits [m] (data_dim: [m]f32) (gis: [m]f32) (his: [m]f32)
                      (l2: f32) (gamma: f32) : (f32, f32, bool) =
  let (missing, rest_points) = partition (\x -> f32.nan == x.1) (zip (iota m) data_dim)
  -- add check empty rest_points
  let missing_gis_sum = reduce (+) 0.0 <| map (\x -> gis[x.0]) missing
  let missing_his_sum = reduce (+) 0.0 <| map (\x -> his[x.0]) missing
  let sorted = radix_sort_float_by_key (.1) f32.num_bits f32.get_bit
                                       rest_points
  let (sorted_idx, sorted_data) = unzip sorted
  let sorted_gis = permute gis sorted_idx
  let sorted_his = permute his sorted_idx
  let scan_gis = scan (+) 0.0 sorted_gis
  let scan_his = scan (+) 0.0 sorted_his
  let gm = last scan_gis + missing_gis_sum
  let hm = last scan_his + missing_his_sum
  let gains = map2 (\g h -> gain g h gm hm l2 gamma missing_gis_sum missing_his_sum)
                   scan_gis scan_his
  let (gains, flags) = unzip gains
  let (best_split_idx, max_gain) = arg_max gains
  let xgboost_split_val = if best_split_idx < (length sorted_idx)-1 then
                            (sorted_data[best_split_idx] + sorted_data[best_split_idx+1])/2
                          else
                            sorted_data[best_split_idx]*2 -- apprently multiply with 2 in xgboost impl
  in
  (xgboost_split_val, max_gain, flags[best_split_idx])

-- (leaf_idx, data_idx, data)
let do_split [n][d][m] (orig_data: [n](i32, i32, [d]f32)) (li: i32) (point_idxs: [m]i32)
                          (points: [m][d]f32) (dim: i32) (value: f32) (missing_flag: bool)
                          : [n](i32, i32, [d]f32) =
  let new_leaf_idx = map (\x ->
                            if x[dim] < value || (f32.nan == x[dim] && missing_flag) then
                              li*2
                            else
                              li*2+1) points
  let new_data = zip3 new_leaf_idx point_idxs points
  in
  -- copy..
  scatter (copy orig_data) point_idxs new_data
  --scatter orig_data point_idxs new_data

let train_round [n][d] (data: [n][d]f32) (labels: [n]f32) (preds: [n]f32) (max_depth: i32) 
                       (l2: f32) (eta: f32) (gamma: f32) : [](i32, f32, bool, bool) =
  let tree = replicate (2**max_depth-1) (0,f32.nan, false, false)
  let gis = map2 (\p y -> gradient_mse p y) preds labels
  let his = map2 (\p y -> hessian_mse p y) preds labels
  let data'= zip3 (replicate n 1) (iota n) data  -- (leaf_id, index to data, data)
  let active_leafs = [1] -- remeber off by one in scatter update to tree
  let (_, r, _) =
    loop (active_leafs, tree, data) = (active_leafs, tree, data') for l < max_depth do
      --let (leaf_min, leaf_max) = if l == 0 then (1,1) else (2**l, 2**(l+1)-1)
      --let new_leafs = replicate 2**(l) scatte leafs or list concat? scatter results in filter!
      let (_, new_tree, data, new_leafs) =
        loop (active_leafs, new_tree, data, new_leafs) = (active_leafs, tree, data, [])
        while !(null active_leafs) do
        let leaf_idx = head active_leafs
        -- add singleton check?
        let (_, point_idxs, data_points) = filter (\x -> x.0 == leaf_idx) data |> unzip3
        in
        if length point_idxs == 1 then -- cannot split node with one ele :) -- min size?
          let point_idx = (head point_idxs)
          let weight = eta*(-gis[point_idx]/(his[point_idx]+l2)) --+ min_weight
          let new_tree = scatter new_tree [leaf_idx-1] [(0, weight, false, false)]

          in
          (tail active_leafs, new_tree, data, new_leafs)
        else
          let gis = permute gis point_idxs
          let his = permute his point_idxs -- scatter conflict?
          let pos_splits = map (\i -> search_splits data_points[:,i] gis his
                                                    l2 gamma) (iota d)
          let (vals, gains, missing_flags) = unzip3 pos_splits
          let (split_dim, gain) = arg_max gains
          let value = vals[split_dim]
          let missing_flag = missing_flags[split_dim]
          let node_flag = gain > 0.0 && l < (max_depth-1)
          let (value, data, new_leafs) =
            if node_flag then
              let ndata = do_split data leaf_idx point_idxs data_points split_dim value missing_flag
              let new_leafs = new_leafs ++ (getChildren leaf_idx)
              in
              (value, ndata, new_leafs)
            else
              let weight = get_leaf_weight gis his l2 eta
              in
              (weight, data, new_leafs)
          let new_tree = scatter new_tree [leaf_idx-1] [(split_dim, value, node_flag, missing_flag)]
          in
          (tail active_leafs, new_tree, data, new_leafs)
      in
      (new_leafs, new_tree, data)
  in
  r


let error [n] (labels: [n]f32) (preds: [n]f32) : f32 =
  let error_sum = reduce (+) 0.0 <| map2 (\l p -> (l-p)**2) labels preds
  in f32.sqrt(error_sum / (f32.i32 n))

let train [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i32) (n_rounds: i32)
                       (l2: f32) (eta: f32) (gamma: f32) : f32 =
  let inital_preds = replicate n 0.5
  let ha = trace n_rounds
  -- scatter to save all trees and return history over n rounds! [n_rounds][2**d](i32,f32,bool)
  -- pad tree before scatter... easiest then simple scatter!
  let res =
    loop (data, labels, preds, e) = (data, labels, inital_preds, 0) for i < n_rounds do
      let tree  = train_round data labels preds max_depth l2 eta gamma --|> trace
      --:[](i32, f32, bool, bool)
      let new_preds = map (\x -> predict x tree) data |> map2 (+) preds --|> trace
      let train_error = error labels new_preds |> trace
      --let ha = trace train_error
      in
      (data, labels, new_preds, train_error)
  in
  res.3
          
let eval = train data[:,:2] data[:,2] 3 20 0.5 0.3 0
--let main : []f32 = map (\i -> train data[:,:2] data[:,2] 2 i 0.5 0.3 0) (iota 20)
