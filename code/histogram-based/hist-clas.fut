import "../objective"
import "bins"
import "histboost"
import "../auc/test"

let train_class [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i64) (n_rounds: i64)
                       (l2: f32) (eta: f32) (gamma: f32) = --: [n_rounds]f32 =
  let inital_preds = replicate n 0.5
  let b = 256
  let (data_b, bin_bounds) = binMap_seq (transpose data) b
  let results = replicate n_rounds 0.0f32

  let (_, error, trees) =
    loop (preds, e, trees) = (inital_preds, results, []) for i < n_rounds do
      let gis = map2 gradient_log preds labels
      let his = map2 hessian_log preds labels
      let tree  = train_round  data_b gis his b max_depth l2 eta gamma |> trace
      --:> [](i64, f32, bool, bool) data
      let new_preds = map (\x -> predict_bin x tree b) data_b |> map2 (+) preds 
      let train_error = auc_score labels new_preds
      let res1 = e with [i] = train_error
      let mapped_tree = map (\x -> let (d, v, flag, miss)= x
                                    let v = if flag then bin_bounds[d, i64.f32 v - 1]
                                            else v
                                    in (d, v, flag, miss)
                             ) tree
      in
      (new_preds, res1, trees ++ mapped_tree)

  in
    error
    --auc_score labels (map (\x -> predict x mapped_tree ) data)
          
let main [n][d] (data: [n][d]f32) (labels: [n]f32) = train_class data labels 6 100 0.5 0.1 0
