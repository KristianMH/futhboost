
let gradient_mse (pred: f32) (orig: f32): f32 = pred-orig

let hessian_mse (pred: f32) (orig: f32): f32 =  1.0


let sigmoid (x: f32) : f32 = f32.exp x/(f32.exp x+1.0)

let gradient_log (pred: f32) (orig: f32) = sigmoid(pred) - orig
--t(ŷ )(1−t(ŷ )).

let hessian_log (pred: f32) (orig: f32) =
  let temp = sigmoid (pred)
  in
  pred*(1.0-pred) -- abs? fmaxf
