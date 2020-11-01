import "../lib/github.com/diku-dk/sorts/radix_sort"
import "../lib/github.com/diku-dk/segmented/segmented"
import "../util"

type binboundaries = (f32, f32) -- min max element in bin


-- assumes vals are sorted!
-- implement so they upper bounds match xgboost technique? n + (n+1) and 2*n at bound
let get_bin_bounds [n] (vals: [n]f32) (b: i64) (n_ele: i64) (rest: i64): [b]binboundaries =
  let bin_sizes = replicate b n_ele
  let lower_bounds_idx = scanExc (+) 0 bin_sizes 
  -- let upper_bounds_idx = rotate 1 lower_bounds_idx 
  -- let upper_bounds_idx = map2 (\l u -> if u <= l then
  --                                        l+n_ele+rest-1 else -- fix for last bin or single_bin
  --                                        u-1) lower_bounds_idx upper_bounds_idx
  let lower_bounds = map (\i -> vals[i]) lower_bounds_idx
  let upper_bounds_idx = rotate 1 lower_bounds_idx
  let upper_bounds = map (\i -> if i==0 then
                                   vals[n-1]*2
                                 else
                                   ((vals[i-1]+vals[i-1])/2.0)) upper_bounds_idx
  in
  zip lower_bounds upper_bounds
  --map2 (\l u-> (vals[l], vals[u])) (lower_bounds_idx) (upper_bounds_idx)
  -- map2 (\l u-> if u == (n-1) then
  --         (vals[l], vals[u]*2.0)
  --       else
  --         (vals[l], (vals[u]+vals[u+1])/2.0)) (lower_bounds_idx) (upper_bounds_idx)
  -- faster with map map zip?


-- handle when n < b ?
-- assumes b > 0
let binMap [n] (vals: [n]f32) (b: i64) : ([n]u16, [b]binboundaries) =
  let dest = replicate n 0u16
  let num_ele_in_bin = n / b
  let rest = n % b
  let index_arr = iota n --|> map u16.i64
  let (s_vals, s_idx) = radix_sort_float_by_key (.0) f32.num_bits f32.get_bit
                                       ( zip vals index_arr) |> unzip
  let val_shape = replicate b num_ele_in_bin with [b-1] = num_ele_in_bin + rest
  let bin_vals = replicated_iota val_shape n |> map u16.i64
  let bin_bounds = get_bin_bounds s_vals b num_ele_in_bin rest
  in
  (scatter dest s_idx bin_vals, bin_bounds)
  -- scatter dest s_idx bin_vals

--let a = [10.3f32, 9.32, 4.32, 3.0, 100.3, 304.3]


-- tests binMap
-- ==
-- entry: binMap_test
-- input { [1.1f32, -3.2, 100.3, 20.3, 10.4, 39.2, 304.3, 7.0, -10.3, 3.3] 3i64}
-- output {[0i64, 0, 2, 2, 1, 2, 2, 1, 0, 1]}
-- input { [1.0f32, 2.3, 42.1, 249.2, -100.0] 2i64}
-- output {[0i64, 1, 1, 1, 0]}
-- input {[10.3f32, 9.32, 4.32, 3.0, 100.3, 304.3] 1i64}
-- output {[0i64, 0, 0, 0, 0, 0]}
entry binMap_test (vals: []f32) (b: i64) =
  (binMap vals b).0

-- ==
-- entry: binMap_test_lower_bounds
-- input { [1.1f32, -3.2, 100.3, 20.3, 10.4, 39.2, 304.3, 7.0, -10.3, 3.3] 3i64}
-- output {[-10.3f32, 3.3, 20.3]}
-- input {[1.0f32, 2.3, 42.1, 249.2, -100.0] 2i64}
-- output {[-100.0f32, 2.3]}
entry binMap_test_lower_bounds (vals: []f32) (b: i64) =
  (binMap vals b).1 |> unzip |> (.0)

-- ==
-- entry: binMap_test_upper_bounds
-- input { [1.1f32, -3.2, 100.3, 20.3, 10.4, 39.2, 304.3, 7.0, -10.3, 3.3] 3i64}
-- output {[1.1f32, 10.4, 304.3]}
-- input {[1.0f32, 2.3, 42.1, 249.2, -100.0] 2i64}
-- output {[1.0f32, 249.2]}
entry binMap_test_upper_bounds (vals: []f32) (b: i64) =
  (binMap vals b).1 |> unzip |> (.1)










-- inplace update wanted to last element instead.
-- let upper_bounds_idx = rotate 1 lower_bounds_idx
--                            with [b-1] = lower_bounds_idx[b-1] + n_ele + rest
--let upper_bounds_idx = map (\t -> t -1) upper_bounds_idx
