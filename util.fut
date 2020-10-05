import "lib/github.com/diku-dk/sorts/radix_sort"
import "lib/github.com/diku-dk/segmented/segmented"
  --(vals, replicate b (0f32, 0f32))
             
                                       






let create_hist [n] (bin_vals: [n]i32) (vals: [n]f32) (num_binds: i32): [num_binds]f32 =
  let dest = replicate num_binds 0.0f32
  in
  reduce_by_index dest (+) 0.0 bin_vals vals


let log2 x = (loop (y,c) = (x,0i32) while y > 1i32 do (y >> 1, c+1)).1

let permute [n][m] 't (xs: [n]t) (idxs: [m]i32): [m]t =
  map (\i -> xs[i]) idxs

let arg_max [n] (xs: [n]f32): (i32,f32) =
    let max ((i1,d1): (i32,f32)) ((i2,d2): (i32,f32)) =
        if d1 > d2 then (i1,d1)
        else if d2 > d1 then (i2,d2)
        else if i1 > i2 then (i1,d1)
        else (i2,d2)
    in reduce_comm max (i32.lowest,f32.lowest) (zip (iota n) xs)
