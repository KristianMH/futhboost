let rightchild 't (i: i32) (tree: []t): (i32, t) =
  (i*2+1, tree[i*2+1])

let leftchild 't (i: i32) (tree: []t): (i32, t) =
  (i*2, tree[i*2])

let parent 't (i: i32) (tree: []t): (i32, t) =
  (i/2, tree[i/2])

let mktree 't (d: i32) (x: t): []t =
  replicate (2**d) x

