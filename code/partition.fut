import "lib/github.com/diku-dk/segmented/segmented"
import "util"

-- change i64 flag_arr to u16 flag_arr unlikely more than 2**16 segs (soft cap)
-- -- -- perserves shape, assumes vals non-empty otherwise pass ne?
-- -- operator: val[i] < limit
-- does apply partition within each segment supplied with shp for a 2D array
-- conds: list of tuples with split dimensions and value for each segment
-- ne: neutral element for scatter
-- op: comparison operator
-- shp: number of elements in each segment
-- vals: values to split
-- Returns: split data and split indicies for each segment.
let partition_lifted [n][l][d] 't (conds: [l](i64, t)) (ne: t) (op: t -> t -> bool) (shp: [l]i64)
                                  (vals: [n][d]t) : ([n][d]t, [l]i64) =
  let flag_arr = mkFlagArray shp 0 1 n
  let bool_flag_arr = map bool.i64 flag_arr
  let seg_offsets_idx = scan (+) 0 flag_arr |> map (\x -> x-1)
  let cs = map2 (\v i -> let (dim, cond_val) = conds[i]
                         in op v[dim] cond_val) vals seg_offsets_idx
  let true_ints = map i64.bool cs
  let false_ints = map (\x -> 1-x) true_ints
  let true_offsets = segmented_scan (+) 0 bool_flag_arr true_ints
  let false_offsets = segmented_scan (+) 0 bool_flag_arr false_ints
  let seg_offsets = scanExc (+) 0 shp
  let num_true_in_segs = segmented_reduce (+) 0 bool_flag_arr true_ints l
  --let num_false_in_segs = map2 (-) shp num_true_in_segs
  let true_val_offsets = map2 (\x i -> x + seg_offsets[i]) true_offsets seg_offsets_idx
  let false_val_offsets = map2 (\x i -> x + seg_offsets[i] + num_true_in_segs[i])
                               false_offsets seg_offsets_idx
  let idxs = map3 (\c iT iF -> if c then iT-1 else iF-1) cs true_val_offsets false_val_offsets
  in
  (scatter2D (replicate n (replicate d ne)) idxs vals, num_true_in_segs)




-- does apply partition within each segment supplied with shp for a 2D array
-- conds: list of tuples with split dimensions and value for each segment
-- ne: neutral element for scatter
-- op: comparison operator
-- shp: number of elements in each segment
-- vals: values to split
-- Returns: index permutation to split data and split indicies for each segment
-- along with boolean values -- should remove boolean values?
let partition_lifted_idx [n][l][d] 't (conds: [l](i64, t)) (ne: t) (op: t -> t -> bool) (shp: [l]i64)
                                  (vals: [n][d]t) : ([n]i64, [l]i64, [n]bool) =
  let flag_arr = mkFlagArray shp 0 1 n
  let bool_flag_arr = map bool.i64 flag_arr
  let seg_offsets_idx = scan (+) 0 flag_arr |> map (\x -> x-1)
  let cs = map2 (\v i -> let (dim, cond_val) = conds[i]
                         in op v[dim] cond_val) vals seg_offsets_idx
  --let ha = trace cs
  let true_ints = map i64.bool cs
  let false_ints = map (\x -> 1-x) true_ints
  let true_offsets = segmented_scan (+) 0 bool_flag_arr true_ints
  let false_offsets = segmented_scan (+) 0 bool_flag_arr false_ints
  let seg_offsets = scanExc (+) 0 shp
  let num_true_in_segs = segmented_reduce (+) 0 bool_flag_arr true_ints l
  --let num_false_in_segs = map2 (-) shp num_true_in_segs
  let true_val_offsets = map2 (\x i -> x + seg_offsets[i]) true_offsets seg_offsets_idx
  let false_val_offsets = map2 (\x i -> x + seg_offsets[i] + num_true_in_segs[i])
                               false_offsets seg_offsets_idx
  let idxs = map3 (\c iT iF -> if c then iT-1 else iF-1) cs true_val_offsets false_val_offsets
  in
  (idxs, num_true_in_segs, cs)

-- let ha =
--   let vals = [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]]
--   let conds = [(1i64, 3), (0i64,4)]
--   let shp = [2i64, 3]
--   in
--   partition_lifted conds 42i32 (<) shp vals |> trace
-- tests partition_lifted
-- ==
-- entry: partition_lifted_test_splits
-- input { [1i64,0] [3,4] [2i64,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[[0,1], [1,10], [3,5], [-3,-4], [100,5]]}
-- input { [1i64,0] [1000,1000] [2i64,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]]}
-- input { [1i64,0] [-1000,-1000] [2i64,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]]}
-- input {[1i64] [3] [8i64] [[10,3], [3,1], [5,2], [10, -3], [30,30], [4,3], [1,1], [3,0]]}
-- output {[[3,1], [5,2], [10,-3], [1,1], [3,0], [10,3], [30, 30], [4,3]]}
entry partition_lifted_test_splits (dims: []i64) (conds: []i32) (shp: []i64)
                                  (vals: [][]i32) =
  (partition_lifted (zip dims conds) 42 (<) shp vals).0
-- missing tests. zero shp however wont be encountered in boosting trees though.
-- ==
-- entry: partition_lifted_test_idxs
-- input { [1i64,0] [3,4] [2i64,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[1i64, 2]}
-- input { [1i64,0] [1000,1000] [2i64,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[2i64, 3]}
-- input { [1i64,0] [-1000,-1000] [2i64,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[0i64,0]}
-- input {[1i64] [3] [8i64] [[10,3], [3,1], [5,2], [10, -3], [30,30], [4,3], [1,1], [3,0]]}
-- output {[5i64]}
entry partition_lifted_test_idxs (dims: []i64) (conds: []i32) (shp: []i64)
                                  (vals: [][]i32) =
  (partition_lifted (zip dims conds) 64 (<) shp vals).1
