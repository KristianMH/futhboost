let rightchild 't (i: i64) (tree: []t): (i64, t) =
  (i*2+1, tree[i*2+1])

let leftchild 't (i: i64) (tree: []t): (i64, t) =
  (i*2, tree[i*2])

let parent 't (i: i64) (tree: []t): (i64, t) =
  (i/2, tree[i/2])

let mktree 't (d: i64) (x: t): *[]t =
  let max_num_nodes = (1 << (d+1)) - 1
  in
  replicate max_num_nodes x



let predict (x: []f32) (tree: [](i64, f32, bool, bool)) : f32 =
  let (_, res, _) =
    loop (i, value, at_node)=(1, 0, true) while at_node do
      let (d, v, flag, missing_flag) = tree[i-1]
      in
      if flag then
        if x[d] < v || (x[d] == f32.nan && missing_flag) then
          (i*2, value, at_node)
        else
          (i*2+1, value, at_node)
      else
        (i, v, flag)
  in
  res
