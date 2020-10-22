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

let get_leaf_weight [n][s] (gis: [n]f32) (his: [n]f32) (shp: [s]i64) (l2: f32) (eta: f32)
                       : [s]f32 =
  let terminal_flag_arr = mkFlagArray shp false true n
  let gissums = segmented_reduce (+) 0f32 terminal_flag_arr gis s
  let hissums = segmented_reduce (+) 0f32 terminal_flag_arr his s
  in
  map2 (\gs hs -> eta*(-gs/(hs+l2))) gissums hissums
       --eta*(-gsum/(hsum + l2))-- + min_weight


let gain (gl: f32) (hl: f32) (g: f32) (h: f32) (l2: f32) (gamma: f32) : (f32, bool) =
  let gr = g-gl
  let hr = h-hl
  in (1/2*(gl**2/(hl+l2)+gr**2/(hr+l2)-g**2/(h+l2)), true)

let getChildren (i: i64): [2]i64 =
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
  let split_val = bin_bounds[best_split_idx] |> (.1) -- max. or min?
  let node_left = (gls[best_split_idx], hls[best_split_idx])
  let node_right = tuple_math (-) (g,h) node_left
  in (best_gain, split_val, flags[best_split_idx], node_left, node_right)


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
                  else if g2 > g2 then y
                  else if d1 > d2 then x
                  else y) d1 d2
  let ne = replicate s (f32.lowest, 0, 0)
  in
  reduce_comm max ne splits --prove if max is commutative
    
-- maps over each dim -> map over each segment, everything should be regular with histograms
-- returns (dim_idx, split_val, is_leaf?, missing_dir, node_left, node_right)
let search_splits_segs [d][s][m] (g_hists: [d][s][m]f32) (h_hists: [d][s][m]f32)
                              (g_node: [s]f32) (h_node: [s]f32)
                              (bin_bounds: [d][m]binboundaries) (l2: f32) (gamma: f32)
                              : [s](i64, f32, bool, bool, node_vals, node_vals) =
  let best_splits_dim =
    map3 (\seg_g_hist seg_h_hist bin_bound -> -- map over each dim
            map4 (\g_hist h_hist g h -> -- map over each segment
                    find_split_hist g_hist h_hist bin_bound g h l2 gamma)
                 seg_g_hist seg_h_hist g_node h_node 
         ) g_hists h_hists bin_bounds :> [d][s](f32, f32, bool, node_vals, node_vals)
  let (gains, split_vals, missing_dirs, left_nodes, right_nodes) = map unzip5 best_splits_dim |> unzip5
  let dim_mat = map (\i -> replicate s i ) (iota d)
  let seg_mat = replicate d (iota s)
  let best_splits = find_best_splits (map3 zip3 gains dim_mat seg_mat)
  
  -- need to add terminal leaf flag but then done.
  in
  map (\(gain, dim_id, seg_id) ->
         if gain > 0.0 then
         let split_val = split_vals[dim_id, seg_id]
         let missing_dir = missing_dirs[dim_id, seg_id]
         let left_node = left_nodes[dim_id, seg_id]
         let right_node = right_nodes[dim_id, seg_id]
         in (dim_id, split_val, missing_dir, false, left_node, right_node)
         else
           (0, 0.0, false, true, (0.0, 0.0), (0.0, 0.0))) best_splits





-- return list of (idx, val) (i32, f32) dim id and split val!
-- maybe flag wether it is end? leaf-weight!!! calculation
-- handle missing values ? f32.nan


-- data layout. [d][n] great for hist calculation but not for partition_lifted
-- perfer [n][d] implement transpose
let train_round [n][d][b] (data: [d][n]i64) (bin_bounds: [d][b]binboundaries)
                             (labels: [n]f32) (preds: [n]f32) (max_depth: i64) 
                             (l2: f32) (eta: f32) (gamma: f32) = -- : [](i32, f32, bool, bool) =
  --let max_num_nodes = (1 << (max_depth+1)) - 1
  let tree = mktree max_depth (0i64,f32.nan, false, false) :> *[](i64, f32, bool, bool)
  let gis = map2 (\p y -> gradient_mse p y) preds labels
  let his = map2 (\p y -> hessian_mse p y) preds labels
  let data_x = data
  -- leaf consist of id, #num_elements and (G, H) sums
  let root = zip3 [1i64] [n] [(0.0, 0.0)]
  -- combine active_leafs and shp? most likely
  let (_, res, _, _, _, _) =  --loop gis and his?
  	-- leafs : [l_shp](i32, i32, (f32,f32)) are the leaves on the previous level
  	-- tree  : [max_num_nodes](i32, f32, bool, bool) is the tree computed so far, updated by scatters
  	-- i : i32 is the current level in the tree
  	-- data : [d][active_points_length]Z_b, active points at the current level in the tree
  	-- gis, his : [active_points_length]f32
    loop (leafs, tree, i, data, gis, his) = -- need active_point_idxs?
      (root, tree, 0, data_x, gis, his)
      while (i < max_depth) && !(null leafs) do
    	let active_points_length = length gis
    	let gis  = (gis  :> [active_points_length]f32)
    	let his  = (his  :> [active_points_length]f32)
    	let data = (data :> [d][active_points_length]i64)

    	let l_shp = length leafs
    	let (shp, active_leafs, GH) = unzip3 (leafs :> [l_shp](i64,i64,(f32,f32)))
        -- leaf data, shp is num points in each seg
    
    	let (GS, HS) = unzip GH -- parent infomation for each segment
    	let flag_arr = mkFlagArray shp 0i64 1i64 active_points_length
    	-- seg_offsets are multiplied with #num_bins to fit flat representation of [#segs][b] hists
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
                ) data ) :> [d]( [l_shp][b]f32, [l_shp][b]f32 ) ) |> unzip

    	let splits = search_splits_segs new_hists_gis new_hists_his GS HS bin_bounds
                                    l2 gamma
    	-- splits should be [l_shp](i64, f32, bool, bool, node_vals, node_vals)
    	-- (dim_idx, split_val, missing_dir, terminal_flag, left_node, right_node)
    	let terminal_flags = map (.3) splits |> map (!) -- flip bool
    	let seg_idxs = scan (+) 0 flag_arr
    	let cs = map (\i -> terminal_flags[i-1]) seg_idxs
    	let (seg_i, shps_permute_idxs) = get_permute_idxs terminal_flags
    	let (split_i, points_idxs) = get_permute_idxs cs
    	-- (partition out terminal leafs on points)
    	let (active_shp, terminal_shp) = permute shp shps_permute_idxs |> split seg_i
    	let (active_leafs, terminal_leafs) = permute active_leafs shps_permute_idxs |> split split_i
    	-- leaf_idxs! match terminal leafs and active leafs for scatter.
    	let (data, _) = permute data points_idxs |> split split_i
    	let (act_arrs, fin_arrs) = permute (zip gis his) points_idxs |> split split_i
    	let (gis,his) = unzip act_arrs
    	let (gis', his') = unzip fin_arrs
    	-- permute vs scatter performance?
        let num_terminal = length terminal_shp
    	let leaf_weights = get_leaf_weight gis' his' terminal_shp l2 eta --:> [num_terminal]f32
        let terminal_leafs_idx = permute terminal_leafs (scanExc (+) 0 terminal_shp)
                                         --:> [num_terminal]i64
    	-- process active_leafs

        let (active_splits, _) = permute splits shps_permute_idxs |> split seg_i
    	-- tree will be updated with scatter -- need to combine for now.
        let terminal_entries = map (\w ->(0i64,w,false, true)) leaf_weights
    	let tree1 = scatter tree (map (\t -> t-1) terminal_leafs_idx) terminal_entries

        let active_leaf_idxs = permute active_leafs (scanExc (+) 0 active_shp)
        let new_nodes = map (\x -> (x.0,x.1,x.2,x.3)) active_splits
    	let tree2 = scatter tree1 (map (\t -> t-1) active_leaf_idxs) new_nodes
        
        -- then partition_lifted can be used with [s](i32, f32) on data (i.e. do_splits)
    	-- new leafs can be calulated from partiton
    	-- should data, gis, his be updated or keep an active_points array
    	-- and then load everything at start of loop?
        let conds = map (\x -> (x.0, x.1)) active_splits
    	let (data, shape) = partition_lifted  conds (0i64) (<) active_shp data
    	let new_shp = calc_new_shape active_shp shape
    	let active_leafs = map getChildren active_leafs |> flatten
    	let left_nodes = map (.4) splits
    	let right_nodes = map (.5) splits
    	let GH = map2 (\ln rn -> [ln, rn]) left_nodes right_nodes |> flatten
    	let leafs = zip3 new_shp active_leafs GH
    	in
    		(leafs, tree2, i+1, data, gis, his)
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
