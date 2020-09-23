let arg_max [n] (xs: [n]f32): (i32,f32) =
    let max ((i1,d1): (i32,f32)) ((i2,d2): (i32,f32)) =
        if d1 > d2 then (i1,d1)
        else if d2 > d1 then (i2,d2)
        else if i1 < i2 then (i1,d1)
        else (i2,d2)
    in reduce_comm max (-1i32,-1f32) (zip (iota n) xs)
