import "lib/github.com/diku-dk/sorts/radix_sort"
import "util"
import "data"
--type dataentry = {}

let gradient_mse (pred: f32) (orig: f32): f32 = 1*(pred-orig)

let hessian_mse 't (pred: t) (orig: t): f32 =  1.0

let cost (gis: []f32) (his: []f32) (lamda: f32): f32 =
  let gsum = reduce (+) 0.0 gis
  let hsum = reduce (+) 0.0 his
  in
  - gsum**2 / (2.0*(hsum + lamda))


let get_leaf_weight [n] (gis: [n]f32) (his: [n]f32) (l2: f32): f32 =
  let gsum = reduce (+) 0.0 gis
  let hsum = reduce (+) 0.0 his
  in
  0.5*(-gsum/(hsum + l2))

let gain (gl: f32) (hl:f32) (g:f32) (h:f32) (l2: f32) =
  let gr = g-gl
  let hr = h-hl
  let cost_left = gl**2/(hl+l2)
  let cost_right = gr**2/(hr+l2)
  let cost_node = g**2/(h+l2)
  in
  1/2*(cost_left + cost_right - cost_node)


-- (leaf_idx, data_idx, data)
let do_split [n][d][b] (orig_data: [n](i32, i32, [d]f32)) (li: i32) (leaf_data: [b](i32, i32, [d]f32))
                    (dim: i32) (value: f32): [n](i32, i32, [d]f32) =
  let (_, data_idx, data) = unzip3 leaf_data
  let new_leaf_idx = map (\data -> if data[dim] <= value then li*2 else li*2+1) data
  let new_data = zip3 new_leaf_idx data_idx data
  in
  -- why copy??!!!
  scatter (copy orig_data) data_idx new_data
  --scatter orig_data data_idx new_data



let getChildren (i: i32): [2]i32 =
  [2*i, 2*i+1]
-- return list of (idx, val) (i32, f32) dim id and split val!
-- maybe flag wether it is end? leaf-weight!!! calculation
-- handle missing values ? f32.nan
let train [n][d] (data: [n][d]f32) (labels: [n]f32) : [](i32, f32, bool) =
  let max_depth= 3
  let inital_preds = replicate n 0.5
  let gis = map2 (\p y -> gradient_mse p y) inital_preds labels
  let his = map2 (\p y -> hessian_mse p y) inital_preds labels
  let data_x = zip3 (replicate n 1) (iota n) data
  let active_leafs = [1]
  let l2 = 0.000001
  let (_, res, _, _) =
    loop (active_leafs, tree, data, leaf_count) = (active_leafs,[], data_x, 0)
    while !(null active_leafs) && (log2 leaf_count <= max_depth) do
      let leaf_idx = head active_leafs
      let points_in_leaf = filter (\x -> x.0 == leaf_idx) data
      let ha = length points_in_leaf |> trace -- skip if length 1 ! should elim div by zero!
      let (_, point_idx, data_points) = unzip3 points_in_leaf
      let ha = trace data_points
      -- let (pos_splits) =
      --   loop splits = [] for i < d do -- can be replaced with map! over (iota d)
      let pos_splits = map (\i -> 
          let data_dim = data_points[:,i]
          let sorted = radix_sort_float_by_key (.1) f32.num_bits f32.get_bit
                                             (zip point_idx data_dim)
          let (sorted_idx, sorted_data) = unzip sorted
          let sorted_gis = map (\i -> gis[i]) sorted_idx
          let sorted_his = map (\i -> his[i]) sorted_idx
          let scan_gis = scan (+) 0.0 sorted_gis
          let scan_his = scan (+) 0.0 sorted_his
          let (gm, hm) = (last scan_gis, last scan_his)
          let gains = map2 (\g h -> gain g h gm hm l2) scan_gis scan_his |> trace
          let (best_split_idx, max_gain) = arg_max gains
          --let ha = trace best_split_idx
          -- check bounds?
          let xgboost_split_val = if best_split_idx < (length sorted_idx)-1 then
                                    (sorted_data[best_split_idx] + sorted_data[best_split_idx+1])/2
                                  else
                                    sorted_data[best_split_idx]
          in
          --splits ++ [(sorted_data[best_split_idx], max_gain)]
          --splits ++ [(xgboost_split_val, max_gain)]
          (xgboost_split_val, max_gain) ) (iota d)
      let (dim, gain) = arg_max (unzip pos_splits).1
      let value = pos_splits[dim].0
      let new_leaf_count = leaf_count +1
      let node_flag = (gain > 0.0) && (log2 new_leaf_count < max_depth)
      let (value, data, new_leafs) =
        if node_flag then
        let new_data = do_split data leaf_idx points_in_leaf dim value
        let new_leafs = (tail active_leafs)  ++  (getChildren leaf_idx)
        in
        (value, new_data, new_leafs)
        else
        let gis = map (\i -> gis[i]) point_idx
        let his = map (\i -> his[i]) point_idx
        let weight = get_leaf_weight gis his l2
        in
        (weight, data, tail active_leafs)
      in
      (new_leafs, tree++[(dim, value, node_flag)], data, new_leaf_count)
  in
  res

let eval = train data[:,:2] data[:,2]
             
let main (xs: [][]f32) = let res = train xs[:,:2] xs[:,2] in res[0].1
