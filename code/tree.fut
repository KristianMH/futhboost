let rightchild 't (i: i32) (tree: []t): (i32, t) =
  (i*2+1, tree[i*2+1])

let leftchild 't (i: i32) (tree: []t): (i32, t) =
  (i*2, tree[i*2])

let parent 't (i: i32) (tree: []t): (i32, t) =
  (i/2, tree[i/2])

let mktree 't (d: i32) (x: t): []t =
  replicate (2**d) x



let predict (x: []f32) (tree: [](i32, f32, bool, bool)) : f32 =
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
        -- +0.5 due to initial pred of 0.5
        (i, v, flag)
  in
  res
