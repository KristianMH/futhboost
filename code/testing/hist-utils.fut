import "../lib/github.com/diku-dk/sorts/radix_sort"
import "../lib/github.com/diku-dk/segmented/segmented"
import "../util" -- possible import clash with segmented_reduce?

type node_vals = (f32, f32) -- gsum, hsum

-- Returns the weight of elements in each node
-- gis: gradients for nodes not splitting at level i
-- his: hessians for nodes not splitting at level i
-- shp: number of elements in each node at level i
-- l2, eta: regulization terms.
-- returns [num_nodes]weights
let get_leaf_weight [n][s] (gis: [n]f32) (his: [n]f32) (shp: [s]i64) (l2: f32) (eta: f32)
                       : [s]f32 =
  let terminal_flag_arr = mkFlagArray shp false true n
  let gissums = segmented_reduce (+) 0f32 terminal_flag_arr gis s
  let hissums = segmented_reduce (+) 0f32 terminal_flag_arr his s
  in
    map2 (\gs hs -> eta*(-gs/(hs+l2))) gissums hissums
    -- + min_weight

-- calculates the gain of a split candidate
-- gl: gradients sum of left
-- hl: hessian sum of left
-- g: gradient sum of node
-- h: hessian sum of node
-- Returns (gain, missing_dir)  
let calc_gain (gl: f32) (hl: f32) (g:f32) (h: f32) (l2: f32) (gamma: f32)
         (missing_gis_sum: f32) (missing_his_sum: f32) (cost_no_split: f32) : (f32, bool) =
  let gr = g-gl-missing_gis_sum
  let hr = h-hl-missing_his_sum
  --let cost_no_split = g**2/(h+l2)
  let left = (gl+missing_gis_sum)**2/(missing_his_sum+hl+l2) + gr**2/(hr+l2) - cost_no_split
  let right = gl**2/(hl+l2)+ (gr+missing_gis_sum)**2/(missing_his_sum+hr+l2) - cost_no_split
  in
    if left >= right then
      (1/2*left-gamma, true)
    else
      (1/2*right-gamma, false)


-- find the best split within a dimension
-- g_hist: gradient sums of elements in each bin
-- h_hist: hessian sums of elements in each bin
-- g: gradient sum of all elements in node
-- h: hessian sum of all elements in node
-- returns (gain, split_val, missing_dir, node_left, node_right)
-- node_left: gradient sum and hessian sum to left child
-- node_right: gradient sum and hessian sum to right child
-- since we are working in bins, the split_val is u16 bin_id
let find_split_hist [m] (g_hist: [m]f32) (h_hist: [m]f32) --(bin_bounds: [m]binboundaries)
                        (l2: f32) (gamma: f32)
                        : (f32, u16, bool) =
  -- if m == 1? handle
  -- missing value sums
  let na_gis_sum = last g_hist
  let na_his_sum = last h_hist
  let n = m-1
  -- possible split_points
  let gls = scan (+) 0.0 (init g_hist) :> [n]f32
  let hls = scan (+) 0.0 (init h_hist) :> [n]f32
  let g = na_gis_sum + last gls
  let h = na_his_sum + last hls
  let cost_no_split = g**2/(h+l2)
  -- calculate quality of splits
  let gains = map2 (\gl hl -> calc_gain gl hl g h l2 gamma na_gis_sum na_his_sum
                              cost_no_split) gls hls 
  let (gains, flags) = unzip gains
  let (best_split_idx, best_gain) = arg_max gains --|> trace
  let missing_flag = flags[best_split_idx]
  in (best_gain, u16.i64 best_split_idx, missing_flag )


-- finds the best split within each node(segment) at level i
-- it reduces over dimensions [d][s](gain, dimension, seg_id)
-- max function takes two lists of split candidates for two features
-- and selects the split candidation with the best gain for each segment
-- selections the one with higest dimensions if equal in gain
-- splits: [d][s](gain, dimension, seg_id)
-- returns [s](gain, dimension, seg_id)
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
  -- if comm removed compiler error occurs

-- much simpler...
let find_best_splits_v1 [d][s] (gains: [s][d]f32) : [s](i64, f32) =
  let idxs = iota d
  let max ((i1,d1): (i64,f32)) ((i2,d2): (i64,f32)) =
        if d1 > d2 then (i1,d1)
        else if d2 > d1 then (i2,d2)
        else if i1 > i2 then (i1,d1)
        else (i2,d2)
   in 
  map (\x -> reduce_comm max (-1, f32.lowest) (zip idxs x)) gains
      -- map arg_max gains
  
-- maps over each dim -> map over each segment, everything should be regular with histograms
-- returns (dim_idx, split_val, is_leaf?, missing_dir, node_left, node_right)

-- search for splits within each feature of all split candidates for each bin
-- g_hist: [d][s][m]f32 gradient sums for each bin in each node in each dimension
-- h_hist: [d][s][m]f32 hessian sums for each bin in each node in each dimension
-- g_node: [s]f32 gradient sums for each node
-- h_node: [s]f32 hessian sums for each node

-- Returns [s](dim_id, bin_split, missing_direction, is_leaf_flag, node_left, node_right)
-- returns for each node split_dimension split bin 
-- node_left: gradient sum and hessian sum to left child
-- node_right: gradient sum and hessian sum to right child
let search_splits_segs [d][s][m] (g_hists: [d][s][m]f32) (h_hists: [d][s][m]f32)
                                 (l2: f32) (gamma: f32)
                                 : [s](i64, u16, bool, bool) =
  let best_splits_dim =
    map2 (\seg_g_hist seg_h_hist -> --bin_bound -> -- map over each dim
            map2 (\g_hist h_hist -> -- map over each segment
                    find_split_hist g_hist h_hist --bin_bound g h l2 gamma)
                                    l2 gamma)
                 seg_g_hist seg_h_hist 
         ) g_hists h_hists :> [d][s](f32, u16, bool)--bin_bounds 
  let (gains, split_vals, missing_dirs ) = map unzip3 best_splits_dim |> unzip3
  --let dim_mat = map (\i -> replicate s i ) (iota d)
  --let seg_mat = replicate d (iota s)
  -- find best splits for each seg(node) in each dim
  --let best_splits = find_best_splits (map3 zip3 gains dim_mat seg_mat) --|> trace
  -- find_best_splits should be [s][d](f32) and then arg max. gives best split and index!
  -- reduces complexity of find_best_splits enourmusly.!!!
  -- need to add terminal leaf flag but then done.
  let best_splits_segmnets = find_best_splits_v1 (transpose gains)
  in
  map (\seg_id ->
         let (dim_id, gain) = best_splits_segmnets[seg_id]
         --let dim_id = u16.i64 dim_id
         in
         if gain > 0.0 then
           let split_val = split_vals[dim_id, seg_id]
           let missing_dir = missing_dirs[dim_id, seg_id]
           in
             (dim_id, split_val, missing_dir, false)
         else
             (0, 0u16, false, true )
      ) (iota s)
  -- map (\(gain, dim_id, seg_id) ->
  --        if (gain > 0.0) then
  --          --let ha = trace gain
  --          let split_val = split_vals[dim_id, seg_id]
  --          let missing_dir = missing_dirs[dim_id, seg_id]
  --          let left_node = left_nodes[dim_id, seg_id]
  --          let right_node = right_nodes[dim_id, seg_id]
  --          in
  --            (dim_id, split_val, missing_dir, false, left_node, right_node)
  --        else
  --          (0, 0u16, false, true, (0.0, 0.0), (0.0, 0.0))
  --     ) best_splits



-- calculates the gradient and hessian sums of the data
-- data: [n][d]u16 bin_id of each feature value of the data
-- gis: [n]f32 gradients for elements
-- his: [n]f32 hessians for elemetns
-- flag_arr: [n]i64 flag_arr representing segment starts -- should be changed to u16 and JIT casted
-- num_segs: number of nodes(segments)
-- num_bins: number of bins
-- returns [d][num_nodes][num_bins] of gradient and hessian sums
let create_histograms [n][d] (data: [n][d]u16) (gis: [n]f32) (his: [n]f32)
                      (flag_arr: [n]u16) (num_segs: i64) (num_bins: i64)
                      : ([d][num_segs][num_bins]f32, [d][num_segs][num_bins]f32) =
  -- flat_offsets for reduce by index
  let seg_offsets = scan (+) 0u16 flag_arr |> map (\x -> x-1) |> map (\x -> (i64.u16 x) *num_bins)
  -- seg_offsets are multiplied with #num_bins to fit flat representation

  in
   map (\dim_bins ->
          let g_hist_entry =  replicate (num_segs*num_bins) 0.0f32
          let h_hist_entry =  replicate (num_segs*num_bins) 0.0f32
          let idxs = map i64.u16 dim_bins |> map2 (+) seg_offsets
          let g_seg_hist = reduce_by_index g_hist_entry (+) 0.0 idxs gis
          let h_seg_hist = reduce_by_index h_hist_entry (+) 0.0 idxs his
          in  ( unflatten num_segs num_bins g_seg_hist
              , unflatten num_segs num_bins h_seg_hist
              )
       ) (transpose data)  --:> [d]( [num_segs][b]f32, [num_segs][b]f32 ) ) |> unzip
   |> unzip



-- special partition2D only returns true values for 2D array
let partition2D_true [n][d] 't (data: [n][d]t) (conds: [n]bool) : [][d]t =
  
  let true_idxs = map i32.bool conds |> scan (+) 0i32 -- most likely fused all together tho
  let num_true = last true_idxs
  in
  if num_true == 0 then
    []
  else
    let idxs = map2 (\c i -> if c then i64.i32 i-1 else -1i64) conds true_idxs
  -- let idxs = filter (.0) (zip conds true_idxs) |> unzip |> (.1) |> map (\x -> x-1)
  -- let l = length idxs
  -- let idxs = idxs :> [l]i32
    let ne = head (head data)
    let num_true = i64.i32 num_true
    --let idxs = scatter (replicate l 1) (map i64.i32 idxs) (iota l)
    in
    --permute2D data idxs
    scatter2D (replicate num_true (replicate d ne)) idxs data
    
