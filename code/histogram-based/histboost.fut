import "../lib/github.com/diku-dk/sorts/radix_sort"
import "../lib/github.com/diku-dk/segmented/segmented"
import "../util"
import "../tree"
import "../partition"

import "hist-utils"
import "../objective"       
import "bins"

-- data_b: data_points where the value has been mapped to bin_id. b-1 bin is nan_values
-- bin_bounds: split_values for tree
-- labels: True labels which is optimized towards
-- preds: predictions after previous iterations
-- max_depth: max depth of the tree
-- l2, eta, gamma: regulazations params
let train_round [n][d] (data: [n][d]u16) (gis: [n]f32) (his: [n]f32) (num_bins: i64)
                       (max_depth: i64) (l2: f32) (eta: f32) (gamma: f32)
                       : [](i64, f32, bool, bool) =
  -- create tree to scatter into. Overhead of O(2**d-1-d)
  let tree = mktree max_depth (0i64,f32.nan, false, false)
  -- nodes consist of id, #num_elements in node
  let root = zip [1i64] [n]
  let (_, res, _, _, _, _) = 
    -- nodes : [l_shp](i64, i64) are the active nodes from previous level
    -- tree  : [max_num_nodes](i64, f32, bool, bool) is the tree computed so far,
    --         updated by scatters
    -- i : i32 is the current level in the tree
    -- data : [active_points_length][d]u16, active points at the current level in the tree
    -- gis, his : [active_points_length]f32 of active points at the current level in the tree
    loop (nodes, tree, i, data, gis, his) = (root, tree, 0, data, gis, his)
    while i <= max_depth && !(null nodes) do
      let active_points_length = length gis
      let gis  = (gis  :> [active_points_length]f32)
      let his  = (his  :> [active_points_length]f32)
      let data = (data :> [active_points_length][d]u16)
      let l_shp = length nodes
      let (nodes, shp) = unzip (nodes :> [l_shp](i64, i64))
      let (new_nodes, new_tree, new_data, new_gis, new_his) =
        -- if last level of tree- all remaining nodes is converted into leafs
        if i == max_depth then
          let leaf_weights = get_leaf_weight gis his shp l2 eta
          let entries = map (\w ->(0i64, w, false, false)) leaf_weights --|> trace
          let idxs = map (\x -> x-1) nodes
          let final_tree = scatter tree idxs entries
          in
            ([], final_tree, [], [], [])
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
          
          let (terminal_shp, terminal_nodes, active_shp, active_splits, active_nodes,
               data, gis, his, gis', his') =
            if and terminal_node_flags then -- all nodes must be split.
              ([], [], shp, splits, nodes, data, gis, his, [], [])
            else if and (map (.3) splits) then -- all nodes are done.
              (shp, nodes, [], [], [], [], [], [], gis, his)
            else
              -- number idxs array. u16 is current max number of nodes.
              --let seg_idxs = scan (+) 0u16 flag_arr
              let seg_idxs = scan (+) 0 flag_arr
              -- create boolean array to remove dead data
              --let cs = map (\i -> let i = i64.u16 (i-1)  in terminal_node_flags[i]) seg_idxs
              let cs = map (\i -> terminal_node_flags[i-1]) seg_idxs
              -- terminal and active shp, node idxs
              let (active, terminal) = partition (\x -> !(x.2).3) (zip3 shp nodes splits)
              
              let (active_shp, active_nodes, active_splits) = unzip3 active
              let (terminal_shp, terminal_nodes, _) = unzip3 terminal
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
              (terminal_shp, terminal_nodes, active_shp, active_splits, active_nodes,
               data, gis, his, gis', his')

          -- get terminal leaf values
    	  let leaf_weights = get_leaf_weight gis' his' terminal_shp l2 eta
          let terminal_entries = map (\w ->(0i64, w, false, false)) leaf_weights
          let tree_terminal = scatter tree (map (\x -> x-1) terminal_nodes) terminal_entries
          -- split values in intermediate tree is bin_id
          let new_entries =
            map (\x -> let (dim_id, bin_id) = (x.0, f32.u16 x.1)
                       let value = bin_id + 1.0
                       in
                         (dim_id, value, x.2, true)
                ) active_splits
          let tree_full = scatter tree_terminal (map (\x -> x-1) active_nodes) new_entries
          -- conditions to split at. +1 as we split on bin_id
          let conds = map (\x -> (x.0, x.1+1)) active_splits
          -- partition_lifted with scatters! faster than permute
          let (new_data, new_gis, new_his, split_shape) =
            partition_lifted_by_vals conds 0u16 (<) active_shp data gis his
          -- number of new nodes is times 2
          let num_nodes = 2* length active_shp
          let new_shp = calc_new_shape active_shp split_shape :> [num_nodes]i64
          let active_nodes = map getChildren active_nodes |> flatten :> [num_nodes]i64

          let new_nodes = zip active_nodes new_shp
          in
              (new_nodes, tree_full, new_data, new_gis, new_his)
      in
        (new_nodes, new_tree, i+1, new_data, new_gis, new_his)
  in
  res
  



