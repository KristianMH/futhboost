import "lib/github.com/diku-dk/sorts/radix_sort"
import "lib/github.com/diku-dk/segmented/segmented"
import "util"
import "hist"
import "data"
import "tree"
import "partition"
import "bins"
import "woop"

type node_vals = (f32, f32) -- gsum, hsum

let gradient_mse (pred: f32) (orig: f32): f32 = pred-orig

let hessian_mse 't (pred: t) (orig: t): f32 =  1.0

let cost (gis: []f32) (his: []f32) (lamda: f32): f32 =
  let gsum = reduce (+) 0.0 gis
  let hsum = reduce (+) 0.0 his
  in
  - gsum**2 / (2.0*(hsum + lamda))

let get_leaf_weight [n][s] (gis: [n]f32) (his: [n]f32) (shp: [s]i64) (l2: f32) (eta: f32)
                       : [s]f32 =
  let terminal_flag_arr = mkFlagArray shp false true n
  let gissums = segmented_reduce (+) 0f32 terminal_flag_arr gis s
  --let ha = trace gissums
  let hissums = segmented_reduce (+) 0f32 terminal_flag_arr his s
  --let ha = trace hissums
  in
  map2 (\gs hs -> eta*(-gs/(hs+l2))) gissums hissums
       --eta*(-gsum/(hsum + l2))-- + min_weight


let calc_gain (gl: f32) (hl: f32) (g: f32) (h: f32) (l2: f32) (gamma: f32) : (f32, bool) =
  --let cost_left = trace (gl**2/(hl+l2))
  let gr = g-gl
  let hr = h-hl
  --let cost_right = trace (gr**2/(hr+l2))
  --let cost_node = trace (g**2/(h+l2))
  --let ha = trace (gamma)
  --let res = trace (1/2*(cost_left + cost_right - cost_node) - gamma)
  in (1/2*(gl**2/(hl+l2)+gr**2/(hr+l2)-g**2/(h+l2) -gamma), true)

let getChildren (i: i64): [2]i64 =
  [2*i, 2*i+1]



-- find the best split within a dimension
-- returns (gain, split_val, missing_dir, node_left, node_right)
let find_split_hist [m] (g_hist: [m]f32) (h_hist: []f32) --(bin_bounds: [m]binboundaries)
                        (g: f32) (h: f32) (l2: f32) (gamma: f32)
                        : (f32, i64, bool, node_vals, node_vals) =
  let gls = scan (+) 0.0 g_hist --|> trace
  let hls = scan (+) 0.0 h_hist
  let gains = map2 (\gl hl -> calc_gain gl hl g h l2 gamma) gls hls 
  let (gains, flags) = unzip gains
  --let ha = trace gains
  let (best_split_idx, best_gain) = arg_max gains --|> trace
  --let split_val = bin_bounds[best_split_idx] |> (.1) -- max. or min?
  let node_left = (gls[best_split_idx], hls[best_split_idx])
  let node_right = tuple_math (-) (g,h) node_left
  in (best_gain, best_split_idx, flags[best_split_idx], node_left, node_right)


-- reduce with map2 should be implemented I think- work in progress
-- input (gain, dim_idx, seg_id)
let find_best_splits [d][s] (splits: [d][s](f32, i64, i64))
                             : [s](f32, i64, i64) =
  let max [s] (d1: [s](f32, i64, i64)) (d2: [s](f32, i64, i64))
                                       : [s](f32, i64, i64) =
    map2 (\x y ->
            let (g1, d1, _) = x
            let (g2, d2, _) = y
            in
            if g1 > g2 then x
                  else if g2 > g1 then y
                  else if d1 > d2 then x
                  else y) d1 d2
  let ne = replicate s (f32.lowest, 0, 0)
  in
  reduce_comm max ne splits --prove if max is commutative
    
-- maps over each dim -> map over each segment, everything should be regular with histograms
-- returns (dim_idx, split_val, is_leaf?, missing_dir, node_left, node_right)
let search_splits_segs [d][s][m] (g_hists: [d][s][m]f32) (h_hists: [d][s][m]f32)
                              (g_node: [s]f32) (h_node: [s]f32)
                              --(bin_bounds: [d][m]binboundaries)
                              (l2: f32) (gamma: f32)
                              : [s](i64, i64, bool, bool, node_vals, node_vals) =
  let best_splits_dim =
    map2 (\seg_g_hist seg_h_hist -> --bin_bound -> -- map over each dim
            map4 (\g_hist h_hist g h -> -- map over each segment
                    find_split_hist g_hist h_hist --bin_bound g h l2 gamma)
                                    g h l2 gamma)
                 seg_g_hist seg_h_hist g_node h_node 
         ) g_hists h_hists :> [d][s](f32, i64, bool, node_vals, node_vals)--bin_bounds 
  let (gains, split_vals, missing_dirs, left_nodes, right_nodes) = map unzip5 best_splits_dim |> unzip5
  --let ha = trace gains
  let dim_mat = map (\i -> replicate s i ) (iota d)
  let seg_mat = replicate d (iota s)
  let best_splits = find_best_splits (map3 zip3 gains dim_mat seg_mat) --|> trace
  
  -- need to add terminal leaf flag but then done.
  in
  map (\(gain, dim_id, seg_id) ->
         if (gain > 0.0) then
           let split_val = split_vals[dim_id, seg_id]
           let missing_dir = missing_dirs[dim_id, seg_id]
           let left_node = left_nodes[dim_id, seg_id]
           let right_node = right_nodes[dim_id, seg_id]
           in
             (dim_id, split_val, missing_dir, false, left_node, right_node)
         else
           (0, 0i64, false, true, (0.0, 0.0), (0.0, 0.0))) best_splits





-- return list of (idx, val) (i32, f32) dim id and split val!
-- maybe flag wether it is end? leaf-weight!!! calculation
-- handle missing values ? f32.nan


-- data layout. [d][n] great for hist calculation but not for partition_lifted
-- perfer [n][d] implement transpose
let train_round [n][d][b] (data_f: [n][d]f32) (data_b: [n][d]i64) (bin_bounds: [d][b]binboundaries)
                          (labels: [n]f32) (preds: [n]f32) (max_depth: i64) 
                          (l2: f32) (eta: f32) (gamma: f32) : [](i64, f32, bool, bool) =
  let tree = mktree max_depth (0i64,f32.nan, false, false) --:> *[](i64, f32, bool, bool)
  let gis = map2 (\p y -> gradient_mse p y) preds labels
  let his = map2 (\p y -> hessian_mse p y) preds labels
  -- leaf consist of id, #num_elements and (G, H) sums
  let root = zip3 [1i64] [n] [(reduce (+) 0 gis, reduce (+) 0 his)]
  -- combine active_leafs and shp? most likely
  let (_, res, _, _,_, _, _) =  --loop gis and his?
  	-- leafs : [l_shp](i32, i32, (f32,f32)) are the leaves on the previous level
  	-- tree  : [max_num_nodes](i32, f32, bool, bool) is the tree computed so far, updated by scatters
  	-- i : i32 is the current level in the tree
  	-- data : [d][active_points_length]Z_b, active points at the current level in the tree
  	-- gis, his : [active_points_length]f32
    loop (leafs, tree, i, data_b, data_f, gis, his) = -- need active_point_idxs?
      (root, tree, 0, data_b, data_f, gis, his)
      while (i <= max_depth) && !(null leafs) do
    	let active_points_length = length gis
    	let gis  = (gis  :> [active_points_length]f32)
    	let his  = (his  :> [active_points_length]f32)
    	let data_b = (data_b :> [active_points_length][d]i64)
        let data_f = (data_f :> [active_points_length][d]f32)

        let l_shp = length leafs
        -- leaf idx, shp is num points in each seg, GH contains parent split info.
    	let (active_leafs, shp, GH) = unzip3 (leafs :> [l_shp](i64,i64,(f32,f32)))
        let ha = trace shp
        let (new_leafs, new_tree, new_data_b, new_data_f, new_gis, new_his) = 
          if i == max_depth then
            let leaf_weights = get_leaf_weight gis his shp l2 eta
            let entries = map (\w ->(0i64, w, false, true)) leaf_weights
            let tree_full = scatter tree (map (\t -> t-1) active_leafs) entries
            in
             ([], tree_full, [], [], [], [])
          else
            let (GS, HS) = unzip GH -- parent infomation for each segment
    	    let flag_arr = mkFlagArray shp 0i64 1i64 active_points_length
            -- seg_offsets are multiplied with #num_bins to fit flat representation
            -- of [#segs][b] hists
    	    -- flatten unflatten are used or not?
    	    let seg_offsets = scan (+) 0i64 flag_arr |> map (\x -> (x-1)*b)
    	    let (new_hists_gis, new_hists_his) =
    	      (( map (\dim_bins -> 
        	        let idxs = map2 (+) seg_offsets dim_bins
                        let g_hist_entry =  replicate (l_shp*b) 0.0f32
                        let h_hist_entry =  replicate (l_shp*b) 0.0f32
                        let g_seg_hist = reduce_by_index g_hist_entry (+) 0.0 idxs gis
                        let h_seg_hist = reduce_by_index h_hist_entry (+) 0.0 idxs his
                        in  ( unflatten l_shp b g_seg_hist
                            , unflatten l_shp b h_seg_hist
                            )
                     ) (transpose data_b) ) :> [d]( [l_shp][b]f32, [l_shp][b]f32 ) ) |> unzip
            --let ha = trace new_hists_gis
    	    let splits = search_splits_segs new_hists_gis new_hists_his GS HS --bin_bounds
                                            l2 gamma --|> trace
            --let ha = trace splits
    	    -- splits should be [l_shp](i64, f32, bool, bool, node_vals, node_vals)
    	    -- (dim_idx, split_val, missing_dir, terminal_flag, left_node, right_node)
    	    let terminal_flags = map (.3) splits |> map (!) -- flip bool
            --let ha = trace terminal_flags
    	    let seg_idxs = scan (+) 0 flag_arr
    	    let cs = map (\i -> terminal_flags[i-1]) seg_idxs
            let (seg_i, shps_permute_idxs) = get_permute_idxs terminal_flags --|> trace
    	    let (split_i, points_idxs) = get_permute_idxs cs
    	    -- (partition out terminal leafs on points)
    	    let (active_shp, terminal_shp) = permute shp shps_permute_idxs
                                             |> split seg_i --|> trace
    	    let (active_leafs, terminal_leafs) = permute active_leafs shps_permute_idxs
                                                 |> split seg_i --|> trace
    	    -- leaf_idxs! match terminal leafs and active leafs for scatter.
    	    let (data_b, _) = permute data_b points_idxs |> split split_i
            let (data_f, _) = permute data_f points_idxs |> split split_i
    	    let (act_arrs, fin_arrs) = permute (zip gis his) points_idxs |> split split_i
            let (active_splits, _) = permute splits shps_permute_idxs |> split seg_i
    	    let (gis, his) = unzip act_arrs
    	    let (gis', his') = unzip fin_arrs
    	    -- permute vs scatter performance?
            --let num_terminal = length terminal_shp
            let num_terminal = length terminal_shp
            let terminal_leafs = (terminal_leafs :> [num_terminal]i64)
    	    let leaf_weights = get_leaf_weight gis' his' terminal_shp l2 eta :> [num_terminal]f32
            -- let ha = trace terminal_leafs
            -- let ha = trace (scanExc (+) 0 terminal_shp)
            -- let terminal_leafs_idx = permute terminal_leafs ha
            --:> [num_terminal]i64
    	    -- process active_leafs

   	    -- tree will be updated with scatter -- need to combine for now.
            let terminal_entries = map (\w ->(0i64, w, false, true)) leaf_weights
            --let ha = trace terminal_entries
    	    let tree_with_end_nodes = scatter tree (map (\t -> t-1) terminal_leafs) terminal_entries
            -- let he = trace active_shp
            -- let ho = trace active_leafs
            -- let ha = trace (scanExc (+) 0 active_shp)
            --let active_leaf_idxs = permute active_leafs ha
            let new_nodes = map (\x ->
                                  let (dim_id, bin_id) = trace (x.0, x.1)
                                  let split_val = if bin_id == (b-1) then
                                                    bin_bounds[dim_id, bin_id] |> (.1)
                                                  else
                                                    bin_bounds[dim_id, bin_id] |> (.1)
                                   -- max val for bounds
                                   in
                                     (x.0, split_val, x.2, x.3)) active_splits
    	    let tree_full = scatter tree_with_end_nodes (map (\t -> t-1) active_leafs) new_nodes   
            -- then partition_lifted can be used with [s](i32, f32) on data (i.e. do_splits)
    	    -- new leafs can be calulated from partiton
    	    -- should data, gis, his be updated or keep an active_points array
    	    -- and then load everything at start of loop?
            let conds = map (\x -> (x.0, x.1+1)) active_splits --|> trace
            -- let conds = map (\x ->
            --                    let (dim_id, bin_id) = (x.0, x.1)
            --                    let split_val = if bin_id == (b-1) then
            --                                      bin_bounds[dim_id, bin_id] |> (.1)
            --                                    else
            --                                      bin_bounds[dim_id, bin_id+1] |> (.0)
            --                    in
            --                      (dim_id, split_val)
            --                 ) active_splits
            --let conds = map (\x -> (x.0, x.1)) new_nodes
    	    let (idxs, shape, test) = partition_lifted_idx  conds (0) (<) active_shp data_b
            --let he = trace shape
            --let ha = map2 (\x c -> if c then x else 0) gis test |> reduce (+) 0 |> trace
            --let he = map2 (\x c -> if !c then x else 0) gis test |> reduce (+) 0 |> trace
            -- let ha = trace (ha + he)
            -- let ho = trace (reduce (+) 0 (map i32.bool test))
            -- let ha = trace shape
            let data_b = scatter2D (replicate n (replicate d 0i64)) idxs data_b
            let data_f = scatter2D (replicate n (replicate d 0f32)) idxs data_f
            let gis = scatter (replicate (length gis) 0.0) idxs gis
            --let wow = trace (reduce (+) 0 gis[:shape[0]])
            let his = scatter (replicate (length gis) 0.0) idxs his
            let new_length = 2*length active_shp
   	    let new_shp = calc_new_shape active_shp shape :> [new_length]i64
            --let he = trace new_shp
    	    let active_leafs = map getChildren active_leafs |> flatten :> [new_length]i64
    	    let left_nodes = map (.4) active_splits
    	    let right_nodes = map (.5) active_splits
    	    let GH = map2 (\ln rn -> [ln, rn]) left_nodes right_nodes
                     |> flatten :> [new_length](f32,f32)
    	    let leafs = zip3 active_leafs new_shp GH
    	    in
    	      (leafs, tree_full, data_b, data_f, gis, his)
        in
          (new_leafs, new_tree, i+1, new_data_b, new_data_f, new_gis, new_his)
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
      let tree  = train_round data data_b bin_bounds labels preds max_depth l2 eta gamma |> trace
                  --:> [](i64, f32, bool, bool)
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
let main [n][d] (data: [n][d]f32) (labels: [n]f32) = train data labels 15 2 0.5 0.3 0

let test = train woopdata wooptarget 3 1 0.5 0.3 0
--let data_test = train data_test[:,:2] data_test[:,2] 3 1 0.5 0.3 0
