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


-- arg_max returns the right most if multiple values
let arg_max [n] (xs: [n]f32): (i32,f32) =
    let max ((i1,d1): (i32,f32)) ((i2,d2): (i32,f32)) =
        if d1 > d2 then (i1,d1)
        else if d2 > d1 then (i2,d2)
        else if i1 > i2 then (i1,d1)
        else (i2,d2)
    in reduce_comm max (i32.lowest,f32.lowest) (zip (iota n) xs)
