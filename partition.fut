import "lib/github.com/diku-dk/segmented/segmented"
import "util"

-- addded resulting length to handle errors
let segmented_reduce [n] 't (op: t -> t -> t) (ne: t)
                            (flags: [n]bool) (as: [n]t) (r: i32) : [r]t =
  -- Compute segmented scan.  Then we just have to fish out the end of
  -- each segment.
  let as' = segmented_scan op ne flags as
  -- Find the segment ends.
  let segment_ends = rotate 1 flags
  -- Find the offset for each segment end.
  let segment_end_offsets = segment_ends |> map i32.bool |> scan (+) 0
  --let num_segments = if n > 0 then last segment_end_offsets else 0
  -- Make room for the final result.  The specific value we write here
  -- does not matter; they will all be overwritten by the segment
  -- ends.
  let scratch = replicate r ne
  -- Compute where to write each element of as'.  Only segment ends
  -- are written.
  let index i f = if f then i-1 else -1
  in scatter scratch (map2 index segment_end_offsets segment_ends) as'  

-- creates flag array with shape defined by aoa_shp and values from aoa_val
-- r is to specify returned length to handle compiler warnings.
let mkFlagArray 't [m] 
            (aoa_shp: [m]i32) (zero: t)       
            (aoa_val: [m]t  ) (r: i32) : [r]t =   
  let shp_rot = map (\i->if i==0 then 0       
                         else aoa_shp[i-1]
                    ) (iota m)
  let shp_scn = scan (+) 0 shp_rot            
  --let aoa_len = shp_scn[m-1]+aoa_shp[m-1] not needed.    
  let shp_ind = map2 (\shp ind ->             
                       if shp==0 then -1      
                       else ind               
                     ) aoa_shp shp_scn        
  in scatter (replicate r zero)        
             shp_ind aoa_val



-- -- perserves shape, assumes vals non-empty otherwise pass ne?
-- operator: val[i] < limit
let partition_lifted [n][l][d] 't (conds: [l](i32, t)) (op: t -> t -> bool) (shp: [l]i32)
                                  (vals: [n][d]t) : ([n][d]t, []i32) = 
  let flag_arr = mkFlagArray shp 0 (replicate l 1) n
  let bool_flag_arr = map bool.i32 flag_arr
  --offset cond to cond val and segment
  let seg_offsets_idx = scan (+) 0 flag_arr |> map (\x -> x-1) 
    let cs = map2 (\v i -> let (dim, cond_val) = conds[i]
                         in !(op v[dim] cond_val)) vals seg_offsets_idx -- apply op
                -- negate used to support < similar to xgboost
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
  -- ne parameter pass to function? not needed.


-- tests partition_lifted
-- ==
-- entry: partition_lifted_test
-- input { [1,0] [3,4] [2,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[[0,1], [1,10], [3,5], [-3,-4], [100,5]]}
-- input { [1,0] [1000,1000] [2,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]]}
-- input { [1,0] [-1000,-1000] [2,3] [[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]] }
-- output {[[1,10], [0, 1], [3, 5], [100, 5], [-3, -4]]}
entry partition_lifted_test (dims: []i32) (conds: []i32) (shp: []i32)
                                  (vals: [][]i32) =
  (partition_lifted (zip dims conds) (<) shp vals).0
-- missing tests. zero shp however wont be encountered in boosting trees though.
-- replicate and tests offsets!
