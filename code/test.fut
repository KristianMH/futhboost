import "hist"
import "woop"
import "bins"
import "partition"
import "data"

-- module hist_i32 = mk_hist i32
-- module type scalar = {
--   type t
--   val +: t -> t -> t
--   val -: t -> t -> t
-- }


-- let ha [n] (xs: [n]scalar) (ys: [n]scalar): [n]scalar =
--   map2 (+) xs ys

-- let wooo =
-- let x = map (\r -> binMap r 10i64) (transpose woopdata)
-- let (data_b, bin_bounds) = unzip x
-- let data_b = transpose data_b
-- let res = map (\r -> r[8] <= 4) data_b
-- in res 
-- let bool_arr = [false,true,false,false,true,true,true,true,true,false,true,true
-- ,true,false,true,false,false,false,true,true,true,true,true,false
-- ,true,false,true,true,false,false,true,true,false,true,true,false
-- ,false,false,false,false,false,true,true,true,false,true,true,true
-- ,true,false,true,false,true,true,true,true,false,true,false,false
-- ,true,false,true,true,true,false,false,true,true,true,true,false
-- ,false,true,false,false,true,true,false,true,false,true,true,true
-- ,true,true,true,false,true,true,true,false,false,true,true,false
-- ,false,false,true,false,false,true,true,true,true,true,true,true
-- ,false,true,true,true,true,false,false,false,false,false,false,false
-- ,true,false,false,false,true,false,true,false,true,false,false,true
-- ,true,true,true,false,true,false,false,false,true,false,false,true
-- ,false,true,false,false,true,false,true,true,false,true,false,false
-- ,true,false,true,false,true,false,true,true,false,true,true,false
-- ,false,false,true,true,false,true,false,true,false,false,true,false
-- ,false,true,false,false,false,false,false,true,false,true,false,false
-- ,true,false,false,false,true,false,true,false,true,true,false,false
-- ,false,false,false,false,false,true,true,true,false,true,true,false
-- ,false,false,false,false,true,false,false,true,true,false,true,true
-- ,true,true,false,false,false,true,false,true,false,true,false,false
-- ,false,true,true,true,true,true,false,true,false,false,false,false
-- ,true,false,false,false,false,true,true,true,true,true,false,true
-- ,true,false,true,false,false,true,false,true,true,false,false,false
-- ,false,true,false,true,false,false,true,true,false,false,true,false
-- ,true,false,false,false,true,true,true,false,true,true,true,true
-- ,false,true,false,false,true,false,true,false,true,false,true,false
-- ,true,false,true,true,false,false,false,false,false,false,false,false
-- ,false,false,false,false,false,true,true,true,false,false,true,true
-- ,false,false,true,false,true,true,true,true,false,false,false,false
-- ,true,true,false,true,true,false,false,true,true,false,true,false
-- ,false,true,false,false,false,false,false,false,false,true,true,false
-- ,true,true,false,false,false,false,false,true,true,true,false,true
-- ,false,true,true,true,false,true,false,true,true,true,false,false
-- ,true,false,false,false,true,true,false,false,true,false,true,true
-- ,false,false,false,true,false,true,false,false,true,true,true,true
-- ,true,false,false,false,false,true,false,true,false,true,true,true
-- ,false,true,true,true,true,false,true,true,false,true]
-- let ha = map3 (\c c1 i -> (c == c1, i)) bool_arr wooo (indices wooo)
-- let hehehe = filter (\x -> !x.0) ha
-- let flat_data = flatten woopdata
-- let ee = map (\x -> (x.1, flat_data[x.1])) hehehe
--[(176i64, -3.9719e-2f32), (255i64, 4.636e-3f32)]
let woo = 
  let x = partition_lifted [(8, -0.003304)] 0.0 (<) [442] woopdata
  in x.1
let ha =
  let x = map (\r -> binMap r 3) (transpose data_test)
  let (t, b) = unzip x
  in
  b
