import "../objective"
import "exact"
import "../auc/test"

-- ==
-- entry: main
-- compiled input @ ../data.gz
                   
let train_class [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i64) (n_rounds: i64)
                       (l2: f32) (eta: f32) (gamma: f32) = --: [n_rounds]f32 =
  let inital_preds = replicate n 0.5
  let results = replicate n_rounds 0.0f32

  let max_num_nodes = (1 << (max_depth+1)) - 1
  let trees = replicate (n_rounds*max_num_nodes) (0i64, f32.nan, false, false)
  let (_, error, trees) =
    loop (preds, e, trees) = (inital_preds, results, trees) for i < n_rounds do
      let gis = map2 gradient_log preds labels
      let his = map2 hessian_log preds labels
      let tree  = train_round  data gis his max_depth l2 eta gamma |> trace
      --:> [](i64, f32, bool, bool) data
      let new_preds = map (\x -> predict x tree) data |> map2 (+) preds 
      let train_error = auc_score labels new_preds
      let res1 = e with [i] = train_error
      let offsets = map (+i*max_num_nodes) (indices tree)    
      let new_trees = scatter trees offsets tree
      in
      (new_preds, res1, new_trees)

  in
    error
    --auc_score labels (map (\x -> predict x mapped_tree ) data)
          
let main [n][d] (data: [n][d]f32) (labels: [n]f32) = train_class data labels 6 10 0.5 0.1 0
