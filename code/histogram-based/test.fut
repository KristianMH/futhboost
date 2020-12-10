import "../lib/github.com/diku-dk/sorts/radix_sort"
import "../lib/github.com/diku-dk/segmented/segmented"
import "../util"
import "../partition"

import "hist-utils"
import "../objective"       
import "bins"
  -- creates a tree with 2^(d+1)-1 nodes with dummy value t
let mktree 't (d: i64) (x: t): *[]t =
  let max_num_nodes = (1 << (d+1)) - 1
  in
  replicate max_num_nodes x
-- data_b: data_points where the value has been mapped to bin_id. b-1 bin is nan_values
-- bin_bounds: split_values for tree
-- labels: True labels which is optimized towards
-- preds: predictions after previous iterations
-- max_depth: max depth of the tree
-- l2, eta, gamma: regulazations params
let train_round [n][d] (data: [n][d]u16) (gis: [n]f32) (his: [n]f32) (num_bins: i64)
                       (max_depth: i64) (l2: f32) (eta: f32) (gamma: f32)
                       : ([](i64, f32, bool, i64), i64) =
  -- create tree to scatter into. 
  let tree = replicate 20000 (0i64,f32.nan, false, -1)
  -- nodes consist of id, #num_elements in node
  let root = [n]
  let (_, res, _, _, _, _, offset) = 
    -- nodes : [l_shp](i64, i64) are the active nodes from previous level
    -- tree  : [max_num_nodes](i64, f32, bool, bool) is the tree computed so far,
    --         updated by scatters
    -- i : i32 is the current level in the tree
    -- data : [active_points_length][d]u16, active points at the current level in the tree
    -- gis, his : [active_points_length]f32 of active points at the current level in the tree
    loop (shp, tree, i, data, gis, his, offset) = (root, tree, 0, data, gis, his, 0)
    while i <= max_depth && !(null shp) do
      let active_points_length = length gis
      let gis  = (gis  :> [active_points_length]f32)
      let his  = (his  :> [active_points_length]f32)
      let data = (data :> [active_points_length][d]u16)
      let l_shp = length shp
      let shp = shp :> [l_shp]i64
      let (new_shp, new_tree, new_data, new_gis, new_his, new_offset) =
        -- if last level of tree- all remaining nodes is converted into leafs
        if i == max_depth then
          let leaf_weights = get_leaf_weight gis his shp l2 eta
          let entries = map (\w ->(0i64, w, false, -1)) leaf_weights --|> trace
          let idxs = indices shp |> map (+offset)
          let final_tree = scatter tree idxs entries
          in
            ([], final_tree, [], [], [], offset+l_shp)
        else
          -- flag_arr for calculating offsets to segmented operations matching number of nodes
          --let flag_arr = mkFlagArray shp 0u16 1u16 active_points_length
          let flag_arr = mkFlagArray shp 0 1 active_points_length
          let (new_hist_gis, new_hist_his) = create_histograms data gis his flag_arr l_shp num_bins
          --let ha = trace new_hist_gis
          let splits = search_splits_segs new_hist_gis new_hist_his l2 gamma
    	  -- splits should be [l_shp](i64, f32, bool, bool)
    	  -- (dim_idx, split_val, missing_dir, terminal_flag)
          let terminal_node_flags = map (.3) splits |> map (!)         
          let idxs = iota l_shp
          let (terminal_shp, active_shp, active_splits, data, gis, his, gis', his',
               term_idxs, act_idxs) =
            if and terminal_node_flags then -- all nodes must be split.
              ([], shp, splits, data, gis, his, [], [], [], idxs)
            else if and (map (.3) splits) then -- all nodes are done.
              (shp, [], [], [], [], [], gis, his, idxs, [])
            else
              -- number idxs array. u16 is current max number of nodes.
              --let seg_idxs = scan (+) 0u16 flag_arr
              let seg_idxs = scan (+) 0 flag_arr
              -- create boolean array to remove dead data
              --let cs = map (\i -> let i = i64.u16 (i-1)  in terminal_node_flags[i]) seg_idxs
              let cs = map (\i -> terminal_node_flags[i-1]) seg_idxs
              -- terminal and active shp, node idxs
              --let (active, terminal) = partition (\x -> !(x.2).3) (zip4 shp nodes splits idxs)
              let (active, terminal) =
                partition (\x -> terminal_node_flags[x.2]) (zip3 shp splits idxs)
              let (active_shp, active_splits, act_idxs) = unzip3 active
              let (terminal_shp, _, term_idxs) = unzip3 terminal
              -- special partition2D which only returns active data points, as terminal is ignored
              let data = partition2D_true data cs
              --let (data, _) = zip data cs |> partition (.1) |> (.0) |> unzip
              let l_act = length data
              -- handle computer warnings
              let data = data :> [l_act][d]u16
              let (act_arrs, fin_arrs) = partition (\x -> x.2) (zip3 gis his cs)
              let (gis, his, _) = unzip3 act_arrs :> ([l_act]f32, [l_act]f32, [l_act]bool)
    	      let (gis', his', _) = unzip3 fin_arrs
              in
              (terminal_shp, active_shp, active_splits,
               data, gis, his, gis', his', term_idxs, act_idxs)
              

          let nodes_to_be_written = replicate l_shp (0i64, f32.nan, false, -1)
          -- get terminal leaf values
    	  let leaf_weights = get_leaf_weight gis' his' terminal_shp l2 eta
          let terminal_entries = map (\w ->(0i64, w, false, -1)) leaf_weights
          let nodes_to_be_written = scatter nodes_to_be_written term_idxs terminal_entries
          -- split values in intermediate tree is bin_id
          let num_nodes_in_level = length nodes_to_be_written
          let new_entries =
            map2 (\x i ->
                   let (dim_id, bin_id) = (x.0, f32.u16 x.1)
                   let value = bin_id + 1.0
                   let child = offset+num_nodes_in_level+i*2
                   in
                         (dim_id, value, x.2, child )
                ) active_splits (indices act_idxs) --:> [num_active](i64, f32, bool, bool)
          let nodes_to_be_written = scatter nodes_to_be_written act_idxs new_entries

          let tree =
            if offset+num_nodes_in_level > length tree then
              scatter (replicate (2*offset) (0, f32.nan, false, -1)) (indices tree) tree
            else
              tree
          let tree_full = scatter tree (idxs |> map (+offset)) nodes_to_be_written
          -- conditions to split at. +1 as we split on bin_id
          let conds = map (\x -> (x.0, x.1+1)) active_splits
          -- partition_lifted with scatters! faster than permute
          let (new_data, new_gis, new_his, split_shape) =
            partition_lifted_by_vals conds 0u16 (<) active_shp data gis his

          let new_shp = calc_new_shape active_shp split_shape
          in
            (new_shp, tree_full, new_data, new_gis, new_his, offset+num_nodes_in_level)
      in
        (new_shp, new_tree, i+1, new_data, new_gis, new_his, new_offset)
  in
  (res[:offset], offset)


-- loops through a tree for an element untill it reaches a leaf
-- x: element to predict value
-- tree: decision tree returned from training
let predict (x: []f32) (tree: [](i64, f32, bool, i64)) (start: i64) : f32 =
  let (_, res, _) =
    loop (i, value, at_node)=(start, 0, true) while at_node do
      let (d, v, missing_flag, child) = tree[i]
      in
      if child >= 0 then
        if x[d] < v || (f32.isnan x[d] && missing_flag) then
          (child, value, at_node)
        else
          (child+1, value, at_node)
      else
        (i, v, false)
  in
  res

-- loops through a tree for an element untill it reaches a leaf
-- x: element to predict value
-- tree: decision tree returned from training
let predict_bin (x: []u16) (tree: [](i64, f32, bool, i64)) (b: i64) (start: i64): f32 =
  let nan_bin = u16.i64 b-1
  let (_, res, _) =
    loop (i, _, at_node)=(start, 0, true) while at_node do
      let (d, v, missing_flag, child) = tree[i]
      
      in
      if child >= 0 then
      let value = f32.u16 x[d]
      in
        if value < v || x[d] == nan_bin && missing_flag then
       --if x[d] < v || (x[d] == f32.nan && missing_flag) then
          (child, value, at_node)
        else
          (child+1, value, at_node)
      else
        (i, v, false)
  in
  res
let predict_all [n][d][l][m] (data: [n][d]f32) (trees: [m](i64,f32,bool,i64)) (offsets: [l]i64)
                             (bias: f32) : [n]f32 =
  let pred_trees = map (\x -> map (\i -> predict x trees i) offsets |> f32.sum) data
  in
  map (+bias) pred_trees

let train_reg [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i64) (n_rounds: i64)
                       (l2: f32) (eta: f32) (gamma: f32) = -- : [n_rounds]f32 =
  let b = 256
  let (data_b, bin_bounds) = binMap_seq (transpose data) b
  --let (data_b, bin_bounds) = binMap_seq_v1 data b

  let inital_preds = replicate n 0.5
  let results = replicate n_rounds 0.0f32
  
  let trees = replicate (10000) (0i64, f32.nan, false, -1)
  
  let offsets = replicate n_rounds 0i64
  let (_, errors, trees, offsets, total) =
    loop (preds, e, trees, offsets, total) =
      (inital_preds, results, trees, offsets, 0) for i < n_rounds do
      let gis = map2 gradient_mse preds labels
      let his = map2 hessian_mse preds labels
      let (tree, offset)  = train_round  data_b gis his b max_depth l2 eta gamma
                               --:> [l](i64, f32, bool, bool) 
      let new_preds = map (\x -> predict_bin x tree b 0) data_b |> map2 (+) preds 
      let train_error = squared_error labels new_preds
      let res1 = e with [i] = train_error
      -- trees not done yet!
      let mapped_tree = map (\x -> let (d, v, miss, flag)= x
                               let v = if flag >= 0 then bin_bounds[d, i64.f32 v - 1]
                                       else v
                               let flag = if flag >=0 then flag + total else flag
                               in (d, v, miss, flag)
                            ) tree

      let offsets1 = offsets with [i]=offset
      let trees = if total+offset > length trees then
                    scatter (replicate (2*total) (0i64, f32.nan, false, -1)) (indices trees) trees
                  else
                    trees
      let offsets_tree = map (+total) (indices mapped_tree)
      let new_trees = scatter trees offsets_tree mapped_tree
      in
      (new_preds, res1, new_trees, offsets1, total + offset)
  let flat_ensemble = trees[:total]
  let offsets = scanExc (+) 0 offsets
  let val_error = predict_all data flat_ensemble offsets 0.5
                  |> squared_error labels
  in
  (last errors, val_error)
  --errors
-- ==
-- entry: main
-- compiled input @ ../data.gz 
let main [n][d] (data: [n][d]f32) (labels: [n]f32) = train_reg data labels 6 500 0.5 0.1 0
