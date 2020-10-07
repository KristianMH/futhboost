import "lib/github.com/diku-dk/sorts/radix_sort"
import "lib/github.com/diku-dk/segmented/segmented"
           

-- exclusive scan
let scanExc 't [n] (op: t->t->t) (ne: t) (arr : [n]t) : [n]t =
  scan op ne <| map (\i -> if i>0 then arr[i-1] else ne) (iota n)

-- log2 of x
let log2 x = (loop (y,c) = (x,0i32) while y > 1i32 do (y >> 1, c+1)).1

-- permutes array
let permute [n][m] 't (xs: [n]t) (idxs: [m]i32): [m]t =
  map (\i -> xs[i]) idxs
-- operator applied elementwise on tuple of length 2
let tuple_math 't (op: t -> t-> t)(n1: (t,t)) (n2: (t,t)) = (op n1.0 n2.0, op n1.1 n2.1)

-- arg_max returns the right most if multiple values
let arg_max [n] (xs: [n]f32): (i32,f32) =
    let max ((i1,d1): (i32,f32)) ((i2,d2): (i32,f32)) =
        if d1 > d2 then (i1,d1)
        else if d2 > d1 then (i2,d2)
        else if i1 > i2 then (i1,d1)
        else (i2,d2)
    in reduce_comm max (i32.lowest,f32.lowest) (zip (iota n) xs)

-- creates flag array with shape defined by shp and values val
-- r is to specify returned length to handle compiler warnings.
let mkFlagArray 't [m] 
            (shp: [m]i32) (zero: t)       
            (flag_val: t) (r: i32) : [r]t =
  let shp_ind = scanExc (+) 0 shp
  let vals = replicate m flag_val
  in
  scatter (replicate r zero) shp_ind vals

-- addded resulting length to handle errors (from lib segmented)
let segmented_reduce [n] 't (op: t -> t -> t) (ne: t)
                            (flags: [n]bool) (as: [n]t) (r: i32) : [r]t =
  -- Compute segmented scan.  Then we just have to fish out the end of
  -- each segment.
  let as' = segmented_scan op ne flags as
  -- Find the segment ends.
  let segment_ends = rotate 1 flags
  -- Find the offset for each segment end.
  let segment_end_offsets = segment_ends |> map i32.bool |> scan (+) 0
  -- Make room for the final result.  The specific value we write here
  -- does not matter; they will all be overwritten by the segment
  -- ends.
  let scratch = replicate r ne
  -- Compute where to write each element of as'.  Only segment ends
  -- are written.
  let index i f = if f then i-1 else -1
  in scatter scratch (map2 index segment_end_offsets segment_ends) as'

-- from segmented lib
let replicated_iota [n] (reps:[n]i32) (r: i32) : [r]i32 =
  let s1 = scan (+) 0 reps
  let s2 = map2 (\i x -> if i==0 then 0 else x)
                (iota n) (rotate (-1) s1)
  let tmp = reduce_by_index (replicate r 0) i32.max 0 s2 (iota n)
  let flags = map (>0) tmp
  in segmented_scan (+) 0 flags tmp

