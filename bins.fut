import "lib/github.com/diku-dk/sorts/radix_sort"
import "lib/github.com/diku-dk/segmented/segmented"
import "util"

type binboundaries = (f32, f32) -- min max element in bin

let replicated_iota [n] (reps:[n]i32) (r: i32) : [r]i32 =
  let s1 = scan (+) 0 reps
  let s2 = map2 (\i x -> if i==0 then 0 else x)
                (iota n) (rotate (-1) s1)
  let tmp = reduce_by_index (replicate r 0) i32.max 0 s2 (iota n)
  let flags = map (>0) tmp
  in segmented_scan (+) 0 flags tmp


-- assumes vals are sorted!
-- implement so they upper bounds match xgboost technique? n + (n+1) and 2*n at bound
let get_bin_bounds [n] (vals: [n]f32) (b: i32) (n_ele: i32) (rest: i32): [b]binboundaries =
  let bin_sizes = replicate b n_ele 
  let lower_bounds_idx = scanExc (+) 0 bin_sizes
  let upper_bounds_idx = rotate 1 lower_bounds_idx
  let upper_bounds_idx = map2 (\l u -> if u < l then
                                         l+n_ele+rest-1 else -- fix off by one for last bin
                                         u-1) lower_bounds_idx upper_bounds_idx
  in
  map2 (\l u-> (vals[l], vals[u])) (lower_bounds_idx) (upper_bounds_idx)
  -- faster with map map zip?


-- handle when n < b ?
let binMap [n] (vals: [n]f32) (b: i32) : ([]i32, [b]binboundaries) =
  let dest = replicate n 0i32
  let num_ele_in_bin = n / b
  let rest = n % b
  let (s_vals, s_idx) = radix_sort_float_by_key (.0) f32.num_bits f32.get_bit
                                       ( zip vals (iota n)) |> unzip
  let val_shape = replicate b num_ele_in_bin with [b-1] = num_ele_in_bin + rest
  let bin_vals = replicated_iota val_shape n -- should be changed to support other idxs?
  let bin_bounds = get_bin_bounds s_vals b num_ele_in_bin rest
  in
  (scatter dest s_idx bin_vals, bin_bounds)
  -- scatter dest s_idx bin_vals



-- tests binMap
-- ==
-- entry: binMap_test
-- input { [1.1f32, -3.2, 100.3, 20.3, 10.4, 39.2, 304.3, 7.0, -10.3, 3.3] 3}
-- output {[0, 0, 2, 2, 1, 2, 2, 1, 0, 1]}
-- input { [1.0f32, 2.3, 42.1, 249.2, -100.0] 2}
-- output {[0, 1, 1, 1, 0]}
entry binMap_test (vals: []f32) (b: i32) =
  (binMap vals b).0
-- also check bounds
  -- inplace update wanted to last element instead.
  -- let upper_bounds_idx = rotate 1 lower_bounds_idx
  --                            with [b-1] = lower_bounds_idx[b-1] + n_ele + rest
  --let upper_bounds_idx = map (\t -> t -1) upper_bounds_idx
