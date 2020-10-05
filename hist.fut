module type scalar = {
  type t
  val +: t -> t -> t
  val -: t -> t -> t
}

local module type hist = {
  type t
  val add [n]: [n]t -> [n]t -> [n]t
  val sub [n]: [n]t -> [n]t -> [n]t
}


module mk_hist (T: scalar): hist with t = T.t = {
  type t = T.t
  
  let add [n] (x: [n]t) (y: [n]t) : [n]t =
    T.(map2 (+) x y)

  let sub [n] (x: [n]t) (y: [n]t) : [n]t =
    T.(map2 (-) x y)

}
-- module hist(M : {
--   type scalar
--   val zero : scalar
--   val add : scalar -> scalar -> scalar
--   val sub : scalar -> scalar -> scalar
-- }) = {
--   open M

--   let histadd [n] (x: [n]scalar) (y: [n]scalar) : [n]scalar =
--     map2 add x y

--   let histsub [n] (x: [n]scalar) (y: [n]scalar) : [n]scalar =
--     map2 sub x y
-- }
