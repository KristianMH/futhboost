import "../lib/github.com/diku-dk/sorts/radix_sort"
import "../lib/github.com/diku-dk/segmented/segmented"
import "../util"
--import "hist"
import "data"
import "../tree"
import "../partition"
import "bins"
import "lightgbmbins"
import "woop"
import "hist-utils"
import "../objective"       



-- data layout. [d][n] great for hist calculation but not for partition_lifted
-- perfer [n][d] implement transpose
-- (data_f: [n][d]f32) 
let train_round [n][d][b] (data_b: [n][d]u16) (bin_bounds: [d][b]f32)
                          (labels: [n]f32) (preds: [n]f32) (max_depth: i64) 
                          (l2: f32) (eta: f32) (gamma: f32) : [](i64, f32, bool, bool) =
  -- create tree to scatter into. Overhead of O(2**d-1-d)
  let tree = mktree max_depth (0i64,f32.nan, false, false)
  -- gradients and hessians
  let gis = map2 (\p y -> gradient_mse p y) preds labels
  let his = map2 (\p y -> hessian_mse p y) preds labels
  -- leaf consist of id, #num_elements and (G, H) sums
  let root = zip3 [1i64] [n] [(reduce (+) 0 gis, reduce (+) 0 his)]
  -- combine active_leafs and shp? most likely
  let (_, res, _, _, _, _) =  --loop gis and his?
  	-- leafs : [l_shp](i32, i32, (f32,f32)) are the leaves on the previous level
  	-- tree  : [max_num_nodes](i32, f32, bool, bool) is the tree computed so far, updated by scatters
  	-- i : i32 is the current level in the tree
  	-- data : [d][active_points_length]Z_b, active points at the current level in the tree
  	-- gis, his : [active_points_length]f32
        --loop (leafs, tree, i, data_b, data_f, gis, his) = -- need active_point_idxs?
        loop (leafs, tree, i, data, gis, his) =
          (root, tree, 0, data_b, gis, his)
          --(root, tree, 0, data_b, data_f, gis, his)
        while (i <= max_depth) && !(null leafs) do
          let active_points_length = length gis
    	  let gis  = (gis  :> [active_points_length]f32)
    	  let his  = (his  :> [active_points_length]f32)
          let data = (data :> [active_points_length][d]u16)
    	  --let data_b = (data_b :> [active_points_length][d]u16)
          --let data_f = (data_f :> [active_points_length][d]f32)

          let l_shp = length leafs
          -- leaf idx, shp is num points in each seg, GH contains parent split info.
    	  let (active_leafs, shp, GH) = unzip3 (leafs :> [l_shp](i64,i64,(f32,f32)))
          --let ha = trace shp
          --let (new_leafs, new_tree, new_data_b, new_data_f, new_gis, new_his) =
          let (new_leafs, new_tree, new_data, new_gis, new_his) = 
            if i == max_depth then
              let leaf_weights = get_leaf_weight gis his shp l2 eta
              let entries = map (\w ->(0i64, w, false, true)) leaf_weights |> trace
              let idxs = map (\x -> x-1) active_leafs
              --let final_tree = scatter tree idxs entries
              in
                ([], tree, [], [], [])
                --([], tree_full, [], [], [], [])
            else
              let (GS, HS) = unzip GH
              let flag_arr = mkFlagArray shp 0u16 1u16 active_points_length
              let (new_hists_gis, new_hists_his) = create_histograms data gis his flag_arr l_shp b
              let splits = search_splits_segs new_hists_gis new_hists_his GS HS
                                              l2 gamma
              let terminal_flags = map (.3) splits |> map (!) -- flip bool
              let seg_idxs = scan (+) 0u16 flag_arr
    	      let cs = map (\i -> let i = i64.u16 i - 1 in terminal_flags[i]) seg_idxs
              let (seg_i, shps_permute_idxs) = get_permute_idxs terminal_flags --|> trace
    	      let (split_i, points_idxs) = get_permute_idxs cs
    	      -- (partition out terminal leafs on points)
              let (active, terminal) = permute (zip shp active_leafs) shps_permute_idxs
                                       |> split seg_i
              let (active_shp, active_leafs) = unzip active
              let (terminal_shp, terminal_leafs) = unzip terminal
    	      -- leaf_idxs! match terminal leafs and active leafs for scatter.
              --let (data, _) = permute2D data_b points_idxs |> split split_i
    	      let (act_arrs, fin_arrs) = permute (zip gis his) points_idxs |> split split_i
              let (active_splits, _) = permute splits shps_permute_idxs |> split seg_i
    	      let (gis, his) = unzip act_arrs
    	      let (gis', his') = unzip fin_arrs
    	      -- permute vs scatter performance?
    	      let leaf_weights = get_leaf_weight gis' his' terminal_shp l2 eta

    	      -- process active_leafs

   	      -- tree will be updated with scatter -- need to combine for now.
              let terminal_entries = map (\w ->(0i64, w, false, true)) leaf_weights
              --let ha = trace terminal_entries
    	      let tree_with_end_nodes = scatter tree (map (\t -> t-1) terminal_leafs) terminal_entries
              let tree_full = tree_with_end_nodes
              in
              (leafs, tree_full, data, gis, his)
              -- let (GS, HS) = unzip GH -- parent infomation for each segment
              -- -- seg_offsets are multiplied with #num_bins to fit flat representation
              -- -- of [#segs][b] hists
    	      -- -- flatten unflatten are used or not?

              -- let flag_arr = mkFlagArray shp 0i64 1i64 active_points_length -- OBS!
              -- let (new_hists_gis, new_hists_his) = create_histograms data gis his flag_arr l_shp b

              
              -- --let ha = trace new_hists_gis
    	      -- let splits = search_splits_segs new_hists_gis new_hists_his GS HS --bin_bounds
              --                                  l2 gamma --|> trace
              -- --let splits = replicate l_shp (0, 0u16, false, false, (0.0, 0.0), (0.0, 0.0))
              -- --let ha = trace splits
    	      -- -- splits should be [l_shp](i64, f32, bool, bool, node_vals, node_vals)
    	      -- -- (dim_idx, split_val, missing_dir, terminal_flag, left_node, right_node)
    	      -- let terminal_flags = map (.3) splits |> map (!) -- flip bool
              -- --let ha = trace terminal_flags
              
    	      -- let seg_idxs = scan (+) 0 flag_arr
    	      -- let cs = map (\i -> terminal_flags[i-1]) seg_idxs
              -- let (seg_i, shps_permute_idxs) = get_permute_idxs terminal_flags --|> trace
    	      -- let (split_i, points_idxs) = get_permute_idxs cs
    	      -- -- (partition out terminal leafs on points)
    	      -- let (active_shp, terminal_shp) = permute shp shps_permute_idxs
              --                                  |> split seg_i --|> trace
    	      -- let (active_leafs, terminal_leafs) = permute active_leafs shps_permute_idxs
              --                                      |> split seg_i --|> trace
    	      -- -- leaf_idxs! match terminal leafs and active leafs for scatter.
              -- let (data, _) = permute data_b points_idxs |> split split_i
    	      -- --let (data_b, _) = permute data_b points_idxs |> split split_i
              -- --let (data_f, _) = permute data_f points_idxs |> split split_i
    	      -- let (act_arrs, fin_arrs) = permute (zip gis his) points_idxs |> split split_i
              -- let (active_splits, _) = permute splits shps_permute_idxs |> split seg_i
    	      -- let (gis, his) = unzip act_arrs
    	      -- let (gis', his') = unzip fin_arrs
    	      -- -- permute vs scatter performance?
              -- --let num_terminal = length terminal_shp
              -- let num_terminal = length terminal_shp
              -- let terminal_leafs = (terminal_leafs :> [num_terminal]i64)
    	      -- let leaf_weights = get_leaf_weight gis' his' terminal_shp l2 eta :> [num_terminal]f32
              -- -- let ha = trace terminal_leafs
              -- -- let ha = trace (scanExc (+) 0 terminal_shp)
              -- -- let terminal_leafs_idx = permute terminal_leafs ha
              -- --:> [num_terminal]i64
    	      -- -- process active_leafs

   	      -- -- tree will be updated with scatter -- need to combine for now.
              -- let terminal_entries = map (\w ->(0i64, w, false, true)) leaf_weights
              -- --let ha = trace terminal_entries
    	      -- let tree_with_end_nodes = scatter tree (map (\t -> t-1) terminal_leafs) terminal_entries
              -- -- let he = trace active_shp
              -- -- let ho = trace active_leafs
              -- -- let ha = trace (scanExc (+) 0 active_shp)
              -- --let active_leaf_idxs = permute active_leafs ha
              -- let new_nodes = map (\x ->
              --                        let (dim_id, bin_id) = trace (x.0, x.1)
              --                        let bin_id = i64.u16 bin_id
              --                        let split_val = if  bin_id == (b-1) then
              --                                          bin_bounds[dim_id, bin_id] --|> (.1)
              --                                        else
              --                                          bin_bounds[dim_id, bin_id] --|> (.1)
              --                      -- max val for bounds
              --                      in
              --                        (x.0, split_val, x.2, x.3)) active_splits
    	      -- let tree_full = scatter tree_with_end_nodes (map (\t -> t-1) active_leafs) new_nodes 
              
              -- -- then partition_lifted can be used with [s](i32, f32) on data (i.e. do_splits)
    	      -- -- new leafs can be calulated from partiton
    	      -- -- should data, gis, his be updated or keep an active_points array
    	      -- -- and then load everything at start of loop?
              -- let conds = map (\x -> (x.0, x.1+1)) active_splits --|> trace
              -- --let conds = map (\x -> (x.0, x.1)) new_nodes
    	      -- let (idxs, shape, test) = partition_lifted_idx  conds (0) (<) active_shp data
              --  let data = scatter2D (replicate (length data) (replicate d 0u16)) idxs data
              -- --let data_f = scatter2D (replicate n (replicate d 0f32)) idxs data_f
              -- let gis = scatter (replicate (length gis) 0.0) idxs gis
              -- --let gis = permute gis idxs
              -- --let wow = trace (reduce (+) 0 gis[:shape[0]])
              -- let his = scatter (replicate (length gis) 0.0) idxs his
              -- --let his = permute his idxs -- likely wrong
              -- let new_length = 2*length active_shp
   	      -- let new_shp = calc_new_shape active_shp shape :> [new_length]i64
              -- --let he = trace new_shp
    	      -- let active_leafs = map getChildren active_leafs |> flatten :> [new_length]i64
    	      -- let left_nodes = map (.4) active_splits
    	      -- let right_nodes = map (.5) active_splits
    	      -- let GH = map2 (\ln rn -> [ln, rn]) left_nodes right_nodes
              --          |> flatten :> [new_length](f32,f32)
    	      -- let leafs = zip3 active_leafs new_shp GH
    	      -- in
              --   (leafs, tree_full, data, gis, his)
    	        --(leafs, tree_full, data_b, data_f, gis, his)
          in
            (new_leafs, new_tree, i+1, new_data, new_gis, new_his)
            --(new_leafs, new_tree, i+1, new_data_b, new_data_f, new_gis, new_his)
  in
    res



let error (label: f32) (pred: f32) : f32 = (label-pred)**2

let train [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i64) (n_rounds: i64)
                       (l2: f32) (eta: f32) (gamma: f32) : [n_rounds]f32 =
  let inital_preds = replicate n 0.5
  let (data_b, bin_bounds) = map (\r -> binMap r 10i64) (transpose data) |> unzip
  let data_b = transpose data_b
  let results = replicate n_rounds 0.0f32
  --let ha = map (\i -> trace i) bin_bounds
  let res =
    loop (data, labels, preds, e) = (data, labels, inital_preds, results) for i < n_rounds do
      let tree  = train_round  data_b bin_bounds labels preds max_depth l2 eta gamma |> trace
                  --:> [](i64, f32, bool, bool) data
      let new_preds = map (\x -> predict x tree) data |> map2 (+) preds 
      let train_error = reduce (+) 0.0 <| map2 (\l p -> error l p) labels new_preds
      let train_error = f32.sqrt (train_error/ (f32.i64 n)) --|> trace
      let res1 = scatter e [i] [train_error]
      --let ha = trace train_error
      in
      (data, labels, new_preds, res1)
  in
  res.3
          
--let eval = train data[:,:2] data[:,2] 3 3 0.5 0.3 0
             
--let main (xs: [][]f32) = let res = train xs[:,:2] xs[:,2] in res[0].1
let main [n][d] (data: [n][d]f32) (labels: [n]f32) = train data labels 3 10 0.5 0.3 0

let test = train woopdata wooptarget 3 2 0.5 0.3 0
--let data_test = train data_test[:,:2] data_test[:,2] 3 1 0.5 0.3 0
