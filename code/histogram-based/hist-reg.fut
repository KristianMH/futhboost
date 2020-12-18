import "../objective"
import "bins"
import "histboost"
import "../tree"

let train_reg [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i64) (n_rounds: i64)
                       (l2: f32) (eta: f32) (gamma: f32) = -- : [n_rounds]f32 =
  let b = 256
  --let (data_b, bin_bounds) = binMap_seq (transpose data) b
  let (data_b, bin_bounds) = binMap_seq_v1 data b

  let inital_preds = replicate n 0.5
  let results = replicate n_rounds 0.0f32
  
  let trees = replicate 20000 (0i64, f32.nan, false, -1)
  
  let offsets = replicate n_rounds 0i64
  let (_, errors, trees, offsets, total) =
    loop (preds, e, trees, offsets, total) =
      (inital_preds, results, trees, offsets, 0) for i < n_rounds do
      let gis = map2 gradient_mse preds labels
      let his = map2 hessian_mse preds labels
      let (tree, offset)  = train_round  data_b gis his b max_depth l2 eta gamma
                               --:> [l](i64, f32, bool, bool) 
      let new_preds = map (\x -> predict_bin x tree b 0) data_b |> map2 (+) preds 
      let train_error = squared_error labels new_preds
      let res1 = e with [i] = train_error
     
      let mapped_tree =
        map (\x -> let (d, v, miss, flag)= x
                   let (v, flag ) = if flag >= 0 then
                                      (bin_bounds[d, i64.f32 v - 1], flag + total)
                                    else (v, flag)
                   in (d, v, miss, flag)
                            ) tree

      let offsets1 = offsets with [i]=offset
      let trees = if total+offset > length trees then
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
  --(last errors, val_error)
  last errors
-- ==
-- entry: main
-- compiled input @ ../data.gz
-- input @ ../data/HIGGS_training
let main [n][d] (data: [n][d]f32) (labels: [n]f32) = train_reg data labels 6 100 0.5 0.1 0



-- let pure_training [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i64) (n_rounds: i64)
--                        (l2: f32) (eta: f32) (gamma: f32) =
--   let inital_preds = replicate n 0.5
--   let b = 256
--   let (data_b, bin_bounds) = binMap_seq (transpose data) b
--   let results = replicate n_rounds 0.0f32
--   let ent = mktree max_depth (0i64, f32.nan, false, false)
--   let l = length ent
--   let ent = ent :> [l](i64, f32, bool, bool)
--   let trees = replicate n_rounds ent
--   let res =
--     loop (trees, preds) = (trees, inital_preds) for i < n_rounds do
--       let gis = map2 gradient_mse preds labels
--       let his = map2 hessian_mse preds labels
--       let tree  = train_round  data_b gis his b max_depth l2 eta gamma
--                                :> [l](i64, f32, bool, bool) 
--       let new_preds = map (\x -> predict_bin x tree b) data_b |> map2 (+) preds 
--       in
--       (scatter2D trees [i] [tree], new_preds)
--   let mapped_trees = map (\t -> map (\x -> let (d, v, flag, miss)= x
--                                            let v = if flag then bin_bounds[d, i64.f32 v - 1]
--                                                    else v
--                                            in (d, v, flag, miss)
--                                     ) t
--                          ) res.0

--   in
--   mapped_trees

-- -- ==
-- -- entry: time_reg
-- -- input @ ../data/HIGGS_training
-- entry time_reg [n][d] (data: [n][d]f32) (labels: [n]f32) = pure_training data labels 6 100 0.5 0.1 0
