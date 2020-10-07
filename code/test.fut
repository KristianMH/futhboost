import "hist"


module hist_i32 = mk_hist i32
module type scalar = {
  type t
  val +: t -> t -> t
  val -: t -> t -> t
}


let ha [n] (xs: [n]scalar) (ys: [n]scalar): [n]scalar =
  map2 (+) xs ys
