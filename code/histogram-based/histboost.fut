import "../lib/github.com/diku-dk/sorts/radix_sort"
import "../lib/github.com/diku-dk/segmented/segmented"
import "../util"
import "../tree"
import "../partition"

import "woop"
import "hist-utils"
import "../objective"       
import "bins"

-- data_b: data_points where the value has been mapped to bin_id. b-1 bin is nan_values
-- bin_bounds: split_values for tree
-- labels: True labels which is optimized towards
-- preds: predictions after previous iterations
-- max_depth: max depth of the tree
-- l2, eta, gamma: regulazations params
let train_round [n][d][b] (data: [n][d]u16) (bin_bounds: [d][b]f32)
                          (labels: [n]f32) (preds: [n]f32) (max_depth: i64) 
                          (l2: f32) (eta: f32) (gamma: f32) : [](i64, f32, bool, bool) =
  -- create tree to scatter into. Overhead of O(2**d-1-d)
  let tree = mktree max_depth (0i64,f32.nan, false, false)
  -- gradients and hessians
  let gis = map2 (\p y -> gradient_mse p y) preds labels
  let his = map2 (\p y -> hessian_mse p y) preds labels
  -- nodes consist of id, #num_elements in node and  gradient and hessian sums for node
  let root = zip3 [1i64] [n] [(reduce (+) 0 gis, reduce (+) 0 his)]
  let (_, res, _, _, _, _) = 
    -- nodes : [l_shp](i32, i32, (f32,f32)) are the active nodes from previous level
    -- tree  : [max_num_nodes](i32, f32, bool, bool) is the tree computed so far,
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
      let (nodes, shp, GH) = unzip3 (nodes :> [l_shp](i64, i64, (f32, f32)))
      let (new_nodes, new_tree, new_data, new_gis, new_his) = 
        if i == max_depth then
          let leaf_weights = get_leaf_weight gis his shp l2 eta
          let entries = map (\w ->(0i64, w, false, false)) leaf_weights --|> trace
          let idxs = map (\x -> x-1) nodes
          let final_tree = scatter tree idxs entries
          in
            ([], final_tree, [], [], [])
        else
          let (GS, HS) = unzip GH
          let flag_arr = mkFlagArray shp 0u16 1u16 active_points_length
          let (new_hist_gis, new_hist_his) = create_histograms data gis his flag_arr l_shp b
          --let ha = trace new_hist_gis
          let splits = search_splits_segs new_hist_gis new_hist_his GS HS l2 gamma
    	  -- splits should be [l_shp](i64, f32, bool, bool, node_vals, node_vals)
    	  -- (dim_idx, split_val, missing_dir, terminal_flag, left_node, right_node)
          let terminal_node_flags = map (.3) splits |> map (!)
          let (terminal_shp, terminal_nodes, active_shp, active_splits, active_nodes,
               data, gis, his, gis', his') =
            if and terminal_node_flags then -- all nodes must be split.
              ([], [], shp, splits, nodes, data, gis, his, [], [])
            else if and (map (.3) splits) then -- all nodes are done.
              (shp, nodes, [], [], [], [], [], [], gis, his)
            else
                 
              let seg_idxs = scan (+) 0u16 flag_arr
              let cs = map (\i -> let i = i64.u16 (i-1)  in terminal_node_flags[i]) seg_idxs
              --let (seg_i, shps_permute_idxs) = get_permute_idxs terminal_node_flags |> trace
    	      --let (split_i, points_idxs) = get_permute_idxs cs
              --let ha = trace (split_i, points_idxs)
              let (active, terminal) = partition (\x -> !(x.2).3) (zip3 shp nodes splits)
               --permute (zip3 shp nodes splits) shps_permute_idxs
               --|> split seg_i
               let (active_shp, active_nodes, active_splits) = unzip3 active
               let (terminal_shp, terminal_nodes, _) = unzip3 terminal
               -- permute is flawed!!, consider using scatter!
               -- let (data, _) = permute2D data points_idxs |> split split_i
    	       -- let (act_arrs, fin_arrs) = permute (zip gis his) points_idxs |> split split_i
               -- let (gis, his) = unzip act_arrs
    	       -- let (gis', his') = unzip fin_arrs
               let data = partition2D_true data cs
               let l_act = length data
               let data = data :> [l_act][d]u16
               let (act_arrs, fin_arrs) = partition (\x -> x.2) (zip3 gis his cs)
               let (gis, his, _) = unzip3 act_arrs :> ([l_act]f32, [l_act]f32, [l_act]bool)
    	       let (gis', his', _) = unzip3 fin_arrs
               in
               (terminal_shp, terminal_nodes, active_shp, active_splits, active_nodes,
                data, gis, his, gis', his')

    	  -- permute vs scatter performance?
    	  let leaf_weights = get_leaf_weight gis' his' terminal_shp l2 eta
          let terminal_entries = map (\w ->(0i64, w, false, false)) leaf_weights
          let tree_terminal = scatter tree (map (\x -> x-1) terminal_nodes) terminal_entries
          -- let ha = trace active_splits
          -- let ha = trace active_shp
          -- let ha = trace terminal_shp
          let new_entries =
            map (\x -> let (dim_id, bin_id) = (x.0, i64.u16 x.1)
                       --let value = bin_bounds[dim_id, bin_id]
                       let value = f32.i64 bin_id
                       in
                         (dim_id, value, x.2, true)
                ) active_splits
          let tree_full = scatter tree_terminal (map (\x -> x-1) active_nodes) new_entries
          let conds = map (\x -> (x.0, x.1+1)) active_splits
          -- let (new_data, new_gis, new_his, split_shape) =
          -- partition_lifted_by_vals conds 0u16 (<) active_shp data gis his
          let (permutation_idx, split_shape) =
            partition_lifted_idx conds (<) active_shp data
          let new_data = permute2D data permutation_idx
          let (new_gis, new_his) = permute (zip gis his) permutation_idx |> unzip
          
          let num_nodes = 2* length active_shp
          let new_shp = calc_new_shape active_shp split_shape :> [num_nodes]i64
          -- let ha = trace new_shp
          -- let ha = trace split_shape
          let active_nodes = map getChildren active_nodes |> flatten :> [num_nodes]i64
          let left_nodes = map (.4) active_splits
    	  let right_nodes = map (.5) active_splits
    	  let GH = map2 (\ln rn -> [ln, rn]) left_nodes right_nodes
                   |> flatten :> [num_nodes](f32,f32)
          let new_nodes = zip3 active_nodes new_shp GH
          in
              (new_nodes, tree_full, new_data, new_gis, new_his)
      in
        (new_nodes, new_tree, i+1, new_data, new_gis, new_his)
  in
  res
  
-- Returns sqaured error between label and prediction
let error (label: f32) (pred: f32) : f32 = (label-pred)**2


let train [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i64) (n_rounds: i64)
                       (l2: f32) (eta: f32) (gamma: f32) = --: [n_rounds]f32 =
  let inital_preds = replicate n 0.5
  --let (data_b, bin_bounds) = map (\r -> binMap r 10i64) (transpose data) |> unzip
  --let data_b = transpose data_b
  let b = 256
  let (data_b, bin_bounds) = binMap_seq (transpose data) b
  let results = replicate n_rounds 0.0f32
  --let ha = map (\i -> trace i) bin_bounds
  let (_, error, tree) =
    loop (preds, e, tree) = (inital_preds, results, []) for i < n_rounds do
      let tree  = train_round  data_b bin_bounds labels preds max_depth l2 eta gamma |> trace
      --:> [](i64, f32, bool, bool) data
      let new_preds = map (\x -> predict_bin x tree b) data_b |> map2 (+) preds 
      let train_error = squared_error labels new_preds
      --let train_error = f32.sqrt (train_error/ (f32.i64 n)) --|> trace
      let res1 = scatter e [i] [train_error]
      --let ha = trace train_error
      in
      (new_preds, res1, tree)
  let mapped_tree = map (\x -> let (d, v, flag, miss)= x
                               let v = bin_bounds[d, i64.f32 v]
                               in (d, v, flag, miss)
                        ) tree
  in
  --unzip4 mapped_tree
  error
  --unzip4 (res.2)
          
let main [n][d] (data: [n][d]f32) (labels: [n]f32) = train data labels 6 100 0.5 0.1 0


      
let test = train woopdata wooptarget 3 1 0.5 0.3 0
--let data_test = train data_test[:,:2] data_test[:,2] 3 1 0.5 0.3 0

