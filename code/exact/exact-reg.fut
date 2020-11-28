import "../objective"
import "exact"

let train_reg [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i64) (n_rounds: i64)
                       (l2: f32) (eta: f32) (gamma: f32) = --: [n_rounds]f32 =
  let inital_preds = replicate n 0.5
  let results = replicate n_rounds 0.0
  let results = replicate n_rounds 0.0f32
  let ent = mktree max_depth (0i64, f32.nan, false, false)
  let l = length ent
  let ent = ent :> [l](i64, f32, bool, bool)
  let trees = replicate n_rounds ent
  
  let (_, error, trees) =
    loop (preds, e, trees) = (inital_preds, results, trees) for i < n_rounds do
      let gis = map2 gradient_mse preds labels
      let his = map2 hessian_mse preds labels
      let tree  = train_round data gis his max_depth l2 eta gamma
                              :> [l](i64, f32, bool, bool)
      let new_preds = map (\x -> predict x tree) data |> map2 (+) preds 
      let train_error = squared_error labels new_preds
      let res1 = e with [i] = train_error
      -- trees not done yet!
      in
      (new_preds, res1, scatter2D trees [i] [tree])
  let predicts = predict_all data trees inital_preds

  in
  (last error, squared_error labels predicts)
          
let main [n][d] (data: [n][d]f32) (labels: [n]f32) = train_reg data labels 6 1 0.5 0.1 0
