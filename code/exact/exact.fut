import "../lib/github.com/diku-dk/sorts/radix_sort"
import "../util"
import "../tree"
import "../objective"
import "../partition"



-- Calculates the weight of a leaf given the list of gradients and
-- hessians given by the equation for optimal weight
let get_leaf_weight [n] (gis: [n]f32) (his: [n]f32) (l2: f32) (eta: f32)
                       : f32 =
  let gsum = reduce (+) 0.0 gis
  let hsum = reduce (+) 0.0 his
  in
  eta*(-gsum/(hsum + l2))-- + min_weight if l1 reg should be implemented.


-- Calculates the gain for a split candidate
-- gl: sum of gradients for points to left
-- hl: sum of hessians for points to left
-- g: sum of gradients for all points in node
-- h: sum of hessians for all points in node
-- missing_gis_sum: sum of gradients for points with NaN value
-- missing_his_sum: sum of hessians for points with NaN value
-- Returns gain and direction for missing values
let gain (gl: f32) (hl: f32) (g:f32) (h: f32) (l2: f32) (gamma: f32)
         (missing_gis_sum: f32) (missing_his_sum: f32) : (f32, bool) =
  let gr = g-gl-missing_gis_sum
  let hr = h-hl-missing_his_sum
  let cost_no_split = g**2/(h+l2)
  let left = (gl+missing_gis_sum)**2/(missing_his_sum+hl+l2) + gr**2/(hr+l2) - cost_no_split
  let right = gl**2/(hl+l2)+ (gr+missing_gis_sum)**2/(missing_his_sum+hr+l2) - cost_no_split
  in
    if left >= right then
      (1/2*left-gamma, true)
    else
      (1/2*right-gamma, false)


-- search find best split candidate within a feature
-- returns best gain, split_value and missing direction
-- data_points: data to seach for split candidate within of feature a
-- gis: gradients to points in data
-- his: hessians to points in data
-- g_node: sum of gradients for all points in node
-- h_node: sum of hessians for all points in node
-- l2,gamma: regulazation terms.
-- returns (gain, split_value, missing_direction)
let search_splits_feature [n] (data_points: [n]f32) (gis: [n]f32) (his: [n]f32)
                          (g_node: f32) (h_node: f32) (l2: f32) (gamma: f32)
                          : (f32, f32, bool) =
  let data = zip3 data_points gis his
  -- split for NaN entries
  let (missing, rest) = partition (\x -> f32.isnan x.0) data
  let (_, miss_g, miss_h) = unzip3 missing
  let missing_gis_sum = reduce (+) 0f32 miss_g
  let missing_his_sum = reduce (+) 0f32 miss_h
  in
  if length rest == 0 then -- only nan values in leaf. cannot split should be node.
    (-1f32, f32.nan, false)
  else
    let (sorted_data, sorted_gis, sorted_his) =
      radix_sort_float_by_key (.0) f32.num_bits f32.get_bit rest |> unzip3
    -- since sorted, find all unique element segments
    let unique_seg_starts = map2 (!=) sorted_data (rotate (-1) sorted_data)
    -- num unique
    let l = map i32.bool unique_seg_starts |> i32.sum |> i64.i32
    in
      if l == 0 then -- all data points have same value cannot split
        (-1f32, f32.nan, false)
      else
      -- sum of gradients and hessians of elements with same value
      -- since we cannot split between elements with same value
      let seg_gis = segmented_reduce (+) 0f32 unique_seg_starts sorted_gis l
      let seg_his = segmented_reduce (+) 0f32 unique_seg_starts sorted_his l
      -- calculate gradient and hessian sums for each possible split
      let scan_gis = scan (+) 0f32 seg_gis
      let scan_his = scan (+) 0f32 seg_his
      -- find unique elements in sorted data for determination of split_value
      let sorted_data = zip sorted_data unique_seg_starts |> filter (.1) |> unzip |> (.0)
      let gains = map2 (\g h -> gain g h g_node h_node l2 gamma missing_gis_sum missing_his_sum)
                       scan_gis scan_his
      let (best_split_idx, max_gain) = (unzip gains).0 |> arg_max
      let xgboost_split_val =
        if best_split_idx < (length sorted_data)-1 then
          (sorted_data[best_split_idx] + sorted_data[best_split_idx+1])/2
        else
          sorted_data[best_split_idx]*2 -- apprently multiply with 2 in xgboost impl
      in
      (max_gain, xgboost_split_val, gains[best_split_idx].1)



-- Does one round of optmizing objective function by finding the tree which minizises it
-- data: input data
-- gis, his: gradient and hessian values for input data
-- max_depth of tree
-- regulazation params l2, eta, gamma. eta is learning rate.
let train_round [n][d] (data: [n][d]f32) (gis: [n]f32) (his: [n]f32) (max_depth: i64) 
                       (l2: f32) (eta: f32) (gamma: f32) : [](i64, f32, bool, bool) =
  let nodes = [1i64]
  let tree = mktree max_depth (0i64, f32.nan, false, false)
  let (final_tree, _, _, _, _, _, _) =
    loop (tree, data, gis, his, i, nodes, shp) =
      (tree, data, gis, his, 0, nodes, [n]) while !(null nodes) && i <= max_depth do
      -- tree: decision tree found so far
      -- data: data points in nodes at level i
      -- gis: gradients in nodes at level i
      -- his: hessians in nodes at level i
      -- nodes: active nodes at level i
      -- shp: number of elements in each node at level i
      let active_points_length = length gis
      let gis  = (gis  :> [active_points_length]f32)
      let his  = (his  :> [active_points_length]f32)
      let data = (data :> [active_points_length][d]f32)

      let num_nodes = length nodes
      let nodes = nodes :> [num_nodes]i64
      let shp = shp :> [num_nodes]i64

      let node_data_offsets = scanExc (+) 0 shp
      let tmp_conds = replicate num_nodes (-1i64, f32.nan)
      -- loops over each node at level i
      -- node_splits are saved on new_conds with (dim, value)
      -- The tree is also updated with the conditation (dim, value, true, missing_dir)
      -- if node should not be split then the "leaf" is written to the tree
      -- leaf is: (0, weight, false, false)
      let (new_tree, new_conds) =
        loop (tree, conds) = (tree, tmp_conds) for j < num_nodes do

          -- node index, offset in data and data, gis his values.
          let node_idx = nodes[j]
          let offset = node_data_offsets[j]
          let num_points_in_node = shp[j]
          let data_in_node = data[offset:offset+num_points_in_node]
          let gis_in_node = gis[offset:offset+num_points_in_node]
          let his_in_node = his[offset:offset+num_points_in_node]
          let g_node = reduce (+) 0f32 gis_in_node
          let h_node = reduce (+) 0f32 his_in_node
          
          in
            if i == max_depth then -- last level of tree i.e. all nodes should be leafs
              let weight = eta*(-g_node/(h_node+l2))
              let leaf = (0i64, weight, false, false)
              let tree = tree with [node_idx-1] = leaf
              in
              (tree, conds) -- no splits since last level
            else
              let pos_splits =
                map (\feature_vals ->
                       search_splits_feature feature_vals gis_in_node his_in_node
                                             g_node h_node l2 gamma
                    ) (transpose data_in_node)
              -- pos_splits is [d](gain, split_value, missing_dir)
              let (gains, vals, dirs) = unzip3 pos_splits
              let (best_split_dim, best_gain) = arg_max gains
              let (node, new_conds) =
                if best_gain > 0f32 then
                  -- split needed. write node into tree and cond for splitting
                  let split_val = vals[best_split_dim]
                  let missing_dir = dirs[best_split_dim]
                  let node = (best_split_dim, split_val, missing_dir, true)
                  let new_conds = conds with [j] = (best_split_dim, split_val)
                  in
                    (node, new_conds)
                else
                  -- no split, so the node is a leaf
                  let weight = eta*(-g_node/(h_node+l2))
                  let leaf = (0i64, weight, false, false)
                  in
                  (leaf, conds)
              -- update tree with node/leaf depending on gain
              let tree = tree with [node_idx-1] = node
              in
              (tree, new_conds)
      -- get active_nodes for splitting data/gis/his 
      let active_node_flags = map (\x -> x.0 >= 0) new_conds
      let (active_nodes, active_shp, active_conds, active_data, active_gis, active_his) =
        if and active_node_flags then -- all splitting
          (nodes, shp, new_conds, data, gis, his)
        else if and (map (!) active_node_flags) then -- no splits needed
          ([], [], [], [], [], [])
        else
           -- find nodes which is split
           let (active_nodes, active_shp, active_conds, _) =
             zip4 nodes shp new_conds active_node_flags |> filter (.3) |> unzip4
                                                                     
           let flag_arr = mkFlagArray shp 0u16 1u16 active_points_length
           let seg_offsets = scan (+) 0u16 flag_arr |> map (\t -> t-1u16)
           -- find active data, gis, his
           -- if filter uses scatter, consider move data into permute2D
           -- where idxs are constructed with scatter albeit in fusion with gis his scatter op?
           -- this not quite optimal yet.
           let (active_data, active_gis, active_his, _) =
             zip4 data gis his seg_offsets |>
             filter (\x -> let idx = i64.u16 x.3 in active_node_flags[idx]) |>
             unzip4
           in
           (active_nodes, active_shp, active_conds, active_data, active_gis, active_his)
      -- lifted partiton permute idxs
      let (permutation_idx, split_shape) =
        partition_lifted_idx active_conds (<) active_shp active_data
      let new_data = permute2D active_data  permutation_idx
      let (new_gis, new_his) = permute (zip active_gis active_his) permutation_idx |> unzip

      let new_shp = calc_new_shape active_shp split_shape 
      -- get node indices for nodes at level i+1
      let new_nodes = map getChildren active_nodes |> flatten
      in
        (new_tree, new_data, new_gis, new_his, i+1, new_nodes, new_shp)
  in
  final_tree

