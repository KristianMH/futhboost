import "lib/github.com/diku-dk/segmented/segmented"
import "util"

-- -- perserves shape, assumes vals non-empty otherwise pass ne?
-- operator: val[i] < limit
let partition_lifted [n][l][d] 't (conds: [l](i32, t)) (op: t -> t -> bool) (shp: [l]i32)
                                  (vals: [n][d]t) : ([n][d]t, []i32) = 
  let flag_arr = mkFlagArray shp 0 1 n
  let bool_flag_arr = map bool.i32 flag_arr
  --offset cond to cond val and segment
  let seg_offsets_idx = scan (+) 0 flag_arr |> map (\x -> x-1) 
  let cs = map2 (\v i -> let (dim, cond_val) = conds[i]
                         in !(op v[dim] cond_val)) vals seg_offsets_idx -- apply op
  -- negate used to support < similar to xgboost i.e true to left
  -- should pass <=
  let true_ints = map i32.bool cs
  let false_ints = map (\x -> 1-x) true_ints
  let true_offsets = segmented_scan (+) 0 bool_flag_arr true_ints
  let false_offsets = segmented_scan (+) 0 bool_flag_arr false_ints |> map (\x -> x-1)
  let seg_offsets = scanExc (+) 0 shp
  let num_true_segs = segmented_reduce (+) 0 bool_flag_arr true_ints l -- #true in each segment
  let num_false_segs = map2 (-) shp num_true_segs -- #false in each segment
  -- off by one to match offset of true
  let num_false_segs_idxs = map (\x -> x-1) num_false_segs 
  let true_val_offsets = map2 (\x i -> x + num_false_segs_idxs[i] +seg_offsets[i])
                              true_offsets seg_offsets_idx 
  let false_val_offsets = map2 (\x i -> x + seg_offsets[i])
                               false_offsets seg_offsets_idx 
  let idxs = map3 (\c iT oT -> if c then iT else oT) cs true_val_offsets false_val_offsets
  in
  (scatter (replicate n vals[0]) idxs vals, num_false_segs)
  -- ne parameter pass to function? not needed as everything get overwritten


-- tests partition_lifted
-- ==
-- entry: partition_lifted_test_splits
-- input { [1,0] [3,4] [2,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[[0,1], [1,10], [3,5], [-3,-4], [100,5]]}
-- input { [1,0] [1000,1000] [2,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]]}
-- input { [1,0] [-1000,-1000] [2,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]]}
-- input {[1] [3] [8] [[10,3], [3,1], [5,2], [10, -3], [30,30], [4,3], [1,1], [3,0]]}
-- output {[[3,1], [5,2], [10,-3], [1,1], [3,0], [10,3], [30, 30], [4,3]]}
entry partition_lifted_test_splits (dims: []i32) (conds: []i32) (shp: []i32)
                                  (vals: [][]i32) =
  (partition_lifted (zip dims conds) (<) shp vals).0
-- missing tests. zero shp however wont be encountered in boosting trees though.
-- ==
-- entry: partition_lifted_test_idxs
-- input { [1,0] [3,4] [2,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[1, 2]}
-- input { [1,0] [1000,1000] [2,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[2, 3]}
-- input { [1,0] [-1000,-1000] [2,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[0,0]}
-- input {[1] [3] [8] [[10,3], [3,1], [5,2], [10, -3], [30,30], [4,3], [1,1], [3,0]]}
-- output {[5]}
entry partition_lifted_test_idxs (dims: []i32) (conds: []i32) (shp: []i32)
                                  (vals: [][]i32) =
  (partition_lifted (zip dims conds) (<) shp vals).1
