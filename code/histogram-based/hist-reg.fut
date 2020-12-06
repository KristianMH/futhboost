import "../objective"
import "bins"
import "histboost"
import "../tree"

-- ==
-- entry: main
-- compiled input @ ../data.gz

let train_reg [n][d] (data: [n][d]f32) (labels: [n]f32) (max_depth: i64) (n_rounds: i64)
                       (l2: f32) (eta: f32) (gamma: f32) = -- : [n_rounds]f32 =
  let b = 256
  --let (data_b, bin_bounds) = binMap_seq (transpose data) b
  let (data_b, bin_bounds) = binMap_seq_v1 data b

  let inital_preds = replicate n 0.5
  let results = replicate n_rounds 0.0f32
  let max_num_nodes = (1 << (max_depth+1)) - 1
  let trees = replicate (n_rounds*max_num_nodes) (0i64, f32.nan, false, false)

  let (_, errors, trees) =
    loop (preds, e, trees) = (inital_preds, results, trees) for i < n_rounds do
      let gis = map2 gradient_mse preds labels
      let his = map2 hessian_mse preds labels
      let tree  = train_round  data_b gis his b max_depth l2 eta gamma
                               --:> [l](i64, f32, bool, bool) 
      let new_preds = map (\x -> predict_bin x tree b) data_b |> map2 (+) preds 
      let train_error = squared_error labels new_preds
      let res1 = e with [i] = train_error
      -- trees not done yet!
      let mapped_tree = map (\x -> let (d, v, flag, miss)= x
                               let v = if flag then bin_bounds[d, i64.f32 v - 1]
                                      else v
                               in (d, v, flag, miss)
                            ) tree
      let offsets = map (+i*max_num_nodes) (indices mapped_tree)    
      let new_trees = scatter trees offsets mapped_tree
      in
      (new_preds, res1, new_trees)
  -- let (_, errors) =
  --   loop (preds, e) = (inital_preds, results) for i < n_rounds do
  --     let gis = map2 gradient_mse preds labels
  --     let his = map2 hessian_mse preds labels
  --     let tree  = train_round  data_b gis his b max_depth l2 eta gamma
  --     let new_preds = map (\x -> predict_bin x tree b) data_b |> map2 (+) preds 
  --     let train_error = squared_error labels new_preds
  --     let res1 = e with [i] = train_error
  --     in
  --     (new_preds, res1)
  in
  errors
          
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
