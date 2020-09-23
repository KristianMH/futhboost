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
  -gsum/(hsum + l2)

let gain (gl: f32) (hl:f32) (g:f32) (h:f32) (l2: f32) =
  let gr = g-gl
  let hr = h-hl
  let cost_left = gl**2/(hl+l2)
  let cost_right = gr**2/(hr+l2)
  let cost_node = g**2/(h+l2)
  in
  1/2*(cost_left + cost_right - cost_node)


-- (leaf_idx, data_idx, data)
let do_split [n][d][b] (orig_data: [n](i32, i32, [d]f32)) (leaf_data: [b](i32, i32, [d]f32))
                    (dim: i32) (value: f32): [n](i32, i32, [d]f32) =
  let (leaf_idxs, data_idx, data) = unzip3 leaf_data
  let new_leaf_idx = map2 (\li data -> if data[dim] <= value then li*2+1 else li*2+2) leaf_idxs data
  let new_data = zip3 new_leaf_idx data_idx data
  in
  -- why copy??!!!
  scatter (copy orig_data) data_idx new_data
  --scatter orig_data data_idx new_data

  
-- return list of (idx, val) (i32, f32) dim id and split val!
-- maybe flag wether it is end? leaf-weight!!! calculation
-- handle missing values ? f32.nan
let train [n][d] (data: [n][d]f32) (labels: [n]f32) : [](i32, f32, bool) =
  let max_depth = 1
  let inital_preds = replicate n 0.5
  let gis = map2 (\p y -> gradient_mse p y) inital_preds labels
  let his = map2 (\p y -> hessian_mse p y) inital_preds labels
  let data_x = zip3 (replicate n 0) (iota n) data
  let l2 = 0
  in
  let (_, res, _) =
    loop (m, tree, data) = (0,[], data_x) while m < max_depth do
      let (_,splits, data) =
        loop (j, splits_found, data) =
          (if m==0 then 0 else 2^(m-1), [], data) while j < 2**m do -- maybe keep list of active leafs?
             let points_in_leaf = filter (\x -> x.0 == j) data
             let (_, point_idx, data_points) = unzip3 points_in_leaf
             let (pos_splits) =
               loop splits = [] for i < d do -- can be replaced with map! over (iota d)
               let data_dim = data_points[:,i]
               let sorted = radix_sort_float_by_key (.1) f32.num_bits f32.get_bit
                                                    (zip point_idx data_dim)
               let (sorted_idx, sorted_data) = unzip sorted
               let sorted_gis = map (\i -> gis[i]) sorted_idx
               let sorted_his = map (\i -> his[i]) sorted_idx
               let scan_gis = scan (+) 0.0 sorted_gis
               let scan_his = scan (+) 0.0 sorted_his
               let (gm, hm) = (last scan_gis, last scan_his)
               let gains = map2 (\g h -> gain g h gm hm l2) scan_gis scan_his
               let (best_split_idx, max_gain) = arg_max gains
               --let ha = trace best_split_idx
               -- check bounds?
               let xgboost_split_val = (sorted_data[best_split_idx] + sorted_data[best_split_idx+1])/2
               in
               --splits ++ [(sorted_data[best_split_idx], max_gain)]
               splits ++ [(xgboost_split_val, max_gain)]
             --let ha = trace pos_splits
             let (dim, gain) = arg_max (unzip pos_splits).1
             let value = pos_splits[dim].0
             let node_flag = (trace gain) > 0.0
             let (value, data) = if node_flag then 
                                 let data = if gain > 0.0 then
                                              do_split data points_in_leaf dim (trace value)
                                            else data
                                 in
                                   (value, data)
                                 else
                                   let gis = map (\i -> gis[i]) point_idx
                                   let his = map (\i -> his[i]) point_idx
                                   in
                                   (get_leaf_weight gis his l2, data)
             in
             (j+1, splits_found ++ [(dim, value, node_flag)], trace data)             
      in
      (m+1, tree++splits, data)
  in
  res

let eval = train data[:,:2] data[:,2]
             
let main (xs: [][]f32) = let res = train xs[:,:2] xs[:,2] in res[0].1
