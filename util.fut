import "lib/github.com/diku-dk/sorts/radix_sort"
import "lib/github.com/diku-dk/segmented/segmented"

type binboundaries = (f32, f32) -- min max element in bin


let segmented_iota [n] (shp: [n]i32) (r: i32): [r]i32  =
  let s1 = scan (+) 0 shp
  let s2 = rotate (-1) s1 with [0] = 0
  let tmp = scatter (replicate r 0 ) s2 (iota n)
  let flag_arr = map (>0) tmp
  in
  segmented_scan (+) 0 flag_arr tmp


let segmented_reduce [n] 't (op: t -> t -> t) (ne: t)
                            (flags: [n]bool) (as: [n]t) (r: i32) : [r]t =
  -- Compute segmented scan.  Then we just have to fish out the end of
  -- each segment.
  let as' = segmented_scan op ne flags as
  -- Find the segment ends.
  let segment_ends = rotate 1 flags
  -- Find the offset for each segment end.
  let segment_end_offsets = segment_ends |> map i32.bool |> scan (+) 0
  let num_segments = if n > 0 then last segment_end_offsets else 0
  -- Make room for the final result.  The specific value we write here
  -- does not matter; they will all be overwritten by the segment
  -- ends.
  let scratch = replicate r ne
  -- Compute where to write each element of as'.  Only segment ends
  -- are written.
  let index i f = if f then i-1 else -1
  in scatter scratch (map2 index segment_end_offsets segment_ends) as'  
  
let scanExc 't [n] (op: t->t->t) (ne: t) (arr : [n]t) : [n]t =
    scan op ne <| map (\i -> if i>0 then arr[i-1] else ne) (iota n)

let segmented_replicate [n] (reps:[n]i32) (vs:[n]i32) (r: i32) : [r]i32 =
  let idxs = segmented_iota reps r
  in map (\i -> vs[i]) idxs

-- assumes vals are sorted!
-- implement so they upper bounds match xgboost technique? n + (n+1) and 2*n at bound
let get_bin_bounds [n] (vals: [n]f32) (b: i32) (n_ele: i32) (rest: i32): [b]binboundaries =
  let ha = replicate b n_ele with [0] = 0
  --let ha[b-1] = n_ele + rest
  let lower_bounds_idx = scan (+) 0 ha
  let upper_bounds_idx = rotate 1 lower_bounds_idx
  let upper_bounds_idx = map2 (\l u -> if u < l then
                                         l+n_ele+rest-1 else
                                         u-1) lower_bounds_idx upper_bounds_idx
  -- inplace update wanted to last element instead.
  -- let upper_bounds_idx = rotate 1 lower_bounds_idx
  --                            with [b-1] = lower_bounds_idx[b-1] + n_ele + rest
  --let upper_bounds_idx = map (\t -> t -1) upper_bounds_idx
  in
  map2 (\l u-> (vals[l], vals[u])) (lower_bounds_idx) (upper_bounds_idx)
  -- faster with map map zip?

let binMap [n] (vals: [n]f32) (b: i32) : ([]i32, [b]binboundaries) =
  let dest = replicate n 0i32
  let num_ele_in_bin = n / b
  let rest = n % b
  let (s_vals, s_idx) = radix_sort_float_by_key (.0) f32.num_bits f32.get_bit
                                       ( zip vals (iota n)) |> unzip
  let val_shape = replicate b num_ele_in_bin with [b-1] = num_ele_in_bin + rest
  let bin_vals = segmented_replicate val_shape (iota b) n -- can be changed to support other idxs
  let bin_bounds = get_bin_bounds s_vals b num_ele_in_bin rest
  in
  (scatter dest s_idx bin_vals, bin_bounds)

  --(vals, replicate b (0f32, 0f32))

let mkFlagArray 't [m] 
            (aoa_shp: [m]i32) (zero: t)       --aoa_shp=[0,3,1,0,4,2,0]
            (aoa_val: [m]t  ) (r: i32) : [r]t =   --aoa_val=[1,1,1,1,1,1,1]
  let shp_rot = map (\i->if i==0 then 0       --shp_rot=[0,0,3,1,0,4,2]
                         else aoa_shp[i-1]
                    ) (iota m)
  let shp_scn = scan (+) 0 shp_rot            --shp_scn=[0,0,3,4,4,8,10]
  let aoa_len = shp_scn[m-1]+aoa_shp[m-1]     --aoa_len= 10
  let shp_ind = map2 (\shp ind ->             --shp_ind= 
                       if shp==0 then -1      --  [-1,0,3,-1,4,8,-1]
                       else ind               --scatter
                     ) aoa_shp shp_scn        --   [0,0,0,0,0,0,0,0,0,0]
  in scatter (replicate r zero)         --   [-1,0,3,-1,4,8,-1]
             shp_ind aoa_val                  --   [1,1,1,1,1,1,1]
                                              -- res = [1,0,0,1,1,0,0,0,1,0]

-- -- perserves shape, assumes vals non-empty
let partition_lifted [n][l] 't (op: t->bool) (shp: [l]i32) (vals: [n]t) : ([n]t, [l]i32) =
  let cs = map op vals
  let true_ints = map i32.bool cs
  let false_ints = map (\x -> 1-x) true_ints
  let flag_arr = mkFlagArray shp false (replicate l true) n
  let true_offsets = segmented_scan (+) 0 flag_arr true_ints
  let false_offsets = segmented_scan (+) 0 flag_arr false_ints |> map (\x -> x-1)
  let seg_offsets = scanExc (+) 0 shp
  let num_true_segs = segmented_reduce (+) 0 flag_arr true_ints l
  let num_false_segs = map2 (-) shp num_true_segs |> map (\x -> x-1)
  let seg_start_offsets = segmented_replicate shp seg_offsets n |> trace
  let true_seg_offsets = segmented_replicate shp num_false_segs n
  let true_val_offsets = map3 (\x y z -> x + y +z) true_seg_offsets true_offsets seg_start_offsets |> trace 
  let false_val_offsets = map2 (+) false_offsets seg_start_offsets |> trace
  let idxs = map3 (\c iT oT -> if c then iT else oT) cs true_val_offsets false_val_offsets
  in
  (scatter (replicate n vals[0]) idxs vals, map2 (-) shp num_true_segs)


let create_hist [n] (bin_vals: [n]i32) (vals: [n]f32) (num_binds: i32): [num_binds]f32 =
  let dest = replicate num_binds 0.0f32
  in
  reduce_by_index dest (+) 0.0 bin_vals vals

let log2 x = (loop (y,c) = (x,0i32) while y > 1i32 do (y >> 1, c+1)).1


let arg_max [n] (xs: [n]f32): (i32,f32) =
    let max ((i1,d1): (i32,f32)) ((i2,d2): (i32,f32)) =
        if d1 > d2 then (i1,d1)
        else if d2 > d1 then (i2,d2)
        else if i1 > i2 then (i1,d1)
        else (i2,d2)
    in reduce_comm max (i32.lowest,f32.lowest) (zip (iota n) xs)
