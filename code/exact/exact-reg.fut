import "../objective"
import "exact"
import "../tree"
       
-- ==
-- entry: main
-- compiled input @ ../data.gz

-- data: 2D traning instances
-- labels: target value to predict
-- max_depth: maximum depth of a signle tree
-- l2, eta, gamma: reguluzation parameters
-- n_rounds: number of boosting rounds
-- Performs gradient boosting using squared error. Returns the final training error.
let boosting_rounds [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i64) (n_rounds: i64)
                       (l2: f32) (eta: f32) (gamma: f32) --: [n_rounds]f32 =
  let inital_preds = replicate n 0.5
  let results = replicate n_rounds 0.0
  -- buffer of 20K entries
  let trees = replicate 20000 (0i64, f32.nan, false, -1)
  
  let offsets = replicate n_rounds 0i64
  let (_, errors, trees, offsets, total) =
    loop (preds, e, trees, offsets, total) =
      (inital_preds, results, trees, offsets, 0) for i < n_rounds do
      let gis = map2 gradient_mse preds labels
      let his = map2 hessian_mse preds labels
      let (tree, offset)  = findOptTree  data gis his max_depth l2 eta gamma

      let new_preds = map (\x -> predict x tree 0) data |> map2 (+) preds 
      let train_error = squared_error labels new_preds
      let res1 = e with [i] = train_error
     
      let mapped_tree =
        map (\x -> let (d, v, miss, flag)= x
                   let (flag) = if flag >= 0 then
                                      flag + total
                                    else flag
                   in (d, v, miss, flag) ) tree

      let offsets1 = offsets with [i]=offset
      let trees =
        if total+offset > length trees then
          scatter (replicate (2*total) (0i64, f32.nan, false, -1)) (indices trees) trees
        else
          trees
      let offsets_tree = map (+total) (indices mapped_tree)
      let new_trees = scatter trees offsets_tree mapped_tree
      in
      (new_preds, res1, new_trees, offsets1, total + offset)
  let flat_ensemble = trees[:total]
  let offsets = scanExc (+) 0 offsets
  let val_error = predict_all data flat_ensemble offsets 0.5
                  |> squared_error labels
  in
  (last errors, val_error)
  --errors
let main [n][d] (data: [n][d]f32) (labels: [n]f32) = boosting_rounds data labels 6 2 0.5 0.1 0
