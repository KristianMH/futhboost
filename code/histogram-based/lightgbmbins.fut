import "../lib/github.com/diku-dk/sorts/radix_sort"
import "../lib/github.com/diku-dk/segmented/segmented"
import "../util"
import "woop"

let zero_thres = 0.00000000000000001f32

let greedyFindBin [l] (distinct_values: [l]f32) (counts: [l]i32)
                      (max_bin: i64) (total_num_samples: i64) (min_data_bin: i64)
                      : ([max_bin]f32, i64) =
  let bin_upper_bounds = replicate max_bin f32.lowest
  let ha = trace (max_bin, l, total_num_samples)
  let offset = 0
  let (bin_upper_bounds, offset) =
    if (l <= max_bin) then
      let (new_bounds, _, new_offset) = 
        loop (upper_bounds, cur_in_bin, offset) = (bin_upper_bounds, 0, offset) for i < l-1 do
          let cur_in_bin = cur_in_bin + counts[i]
          let (new_bounds, new_cur, new_offset) = 
            if (cur_in_bin >= i32.i64 min_data_bin) then
              let value = (distinct_values[i]+distinct_values[i+1])/2f32
              in
                if (offset == 0) || (upper_bounds[offset] <= value) then
                  let cur_in_bin = 0
                  in
                    (upper_bounds with [offset] = value, cur_in_bin, offset+1)
                else
                  (upper_bounds, cur_in_bin, offset)
            else
              (upper_bounds, cur_in_bin, offset)
          in
          (new_bounds, new_cur, new_offset)
      in
      (new_bounds with [new_offset]= f32.highest, new_offset+1)
    else
      let max_bin = if min_data_bin > 0 then
                    let max_bin = i64.min max_bin total_num_samples/min_data_bin
                    in
                      i64.max max_bin 1
                    else
                      max_bin
      let ha = trace max_bin
      let mean_bin_size = f32.i64 total_num_samples / f32.i64 max_bin
      let ha = trace mean_bin_size
      let rest_bin_cnt = max_bin
      let rest_sample_cnt = total_num_samples
      let is_big_count = replicate l false
      let big_offset = 0
      let (is_big_count, big_offset, rest_bin_cnt, rest_sample_cnt) =
        loop (is_big_count, big_offset, rest_bin_cnt, rest_sample_cnt) =
          (is_big_count, big_offset, rest_bin_cnt, rest_sample_cnt) for i < l do
            if (f32.i32 counts[i] >= mean_bin_size) then
              let new_arr = is_big_count with [big_offset] = true
              let new_sample_cnt = rest_sample_cnt- i64.i32 counts[i]
              in
                (new_arr, big_offset+1, rest_bin_cnt-1, new_sample_cnt)
            else
              (is_big_count, big_offset, rest_bin_cnt, rest_sample_cnt)
      --let is_big_count = is_big_count[:big_offset]
      --let ha = trace is_big_count
      let mean_bin_size = f32.i64 rest_sample_cnt / f32.i64 rest_bin_cnt
      let upper_bounds = replicate max_bin f32.highest
      let lower_bounds = replicate max_bin f32.highest
      let bin_cnt = 0
      let lower_bounds = lower_bounds with [bin_cnt] = head distinct_values
      let (upper_bounds, lower_bounds, bin_cnt, _, _, _, _) =
        loop (upper_bounds, lower_bounds, bin_cnt, rest_sample_cnt,
              cur_in_bin, mean_bin_size, rest_bin_cnt) =
          (upper_bounds, lower_bounds, bin_cnt, rest_sample_cnt, 0i32, mean_bin_size, rest_bin_cnt)
          for i < l-1 do
            let rest_sample_cnt = if (!is_big_count[i]) then
                                    rest_sample_cnt- i64.i32 counts[i]
                                  else
                                    rest_sample_cnt
            let cur_in_bin = if (bin_cnt >= max_bin-1) then cur_in_bin else
                               cur_in_bin + counts[i]
            --let ha = trace (cur_in_bin, bin_cnt, mean_bin_size)
            let (upper_bounds,lower_bounds, bin_cnt, cur_in_bin, mean_bin_size, rest_bin_cnt) =
              if is_big_count[i] || (f32.i32 cur_in_bin >= mean_bin_size) ||
                 (is_big_count[i+1] && f32.i32 cur_in_bin >= f32.max 1f32 (mean_bin_size*0.5)) then
                if (bin_cnt == max_bin-2) then -- last update then "break" by just looping no-ops.
                  let upper_bounds = upper_bounds with [bin_cnt] = distinct_values[i]
                  let bin_cnt = bin_cnt+1
                  --let ha = trace (bin_cnt, lower_bounds)
                  let lower_bounds = lower_bounds with [bin_cnt] = distinct_values[i+1]
                  --let ha = trace (lower_bounds, upper_bounds)
                  in
                    (upper_bounds, lower_bounds, bin_cnt, cur_in_bin, mean_bin_size, rest_bin_cnt)
                  else if (bin_cnt < max_bin-1) then
                       let upper_bounds = upper_bounds with [bin_cnt] = distinct_values[i]
                       let bin_cnt = bin_cnt+1
                       let lower_bounds = lower_bounds with [bin_cnt] = distinct_values[i+1]
                       let cur_in_bin = 0
                       let (rest_bin_cnt, mean_bin_size) =
                         if !is_big_count[i] then
                           (rest_bin_cnt-1, f32.i64 rest_sample_cnt / f32.i64 (rest_bin_cnt-1))
                         else
                           (rest_bin_cnt, mean_bin_size)
                       in
                       (upper_bounds, lower_bounds, bin_cnt, cur_in_bin, mean_bin_size, rest_bin_cnt)
                  else
                    (upper_bounds, lower_bounds, bin_cnt, cur_in_bin, mean_bin_size, rest_bin_cnt)
                    -- do nothing
              else
                (upper_bounds, lower_bounds, bin_cnt, 
                 cur_in_bin, mean_bin_size, rest_bin_cnt) -- do notthing
            in
            (upper_bounds, lower_bounds, bin_cnt, rest_sample_cnt,
             cur_in_bin, mean_bin_size, rest_bin_cnt)
      let ha = trace lower_bounds
      let he = trace upper_bounds
      let bin_cnt = bin_cnt +1
      let (bin_upper_bounds, offset) =
        loop (bin_upper_bounds, offset) = (bin_upper_bounds, 0) for i < bin_cnt - 1 do
          let value = (upper_bounds[i] + lower_bounds[i+1])/2f32
          in
            if (offset == 0) || !(bin_upper_bounds[offset] >= value) then -- maybe just >?
              (bin_upper_bounds with [offset] = value, offset+1)
            else
              (bin_upper_bounds, offset)
      --let ha = trace (offset, max_bin, bin_upper_bounds)
      let bin_upper_bounds = bin_upper_bounds with [offset] = f32.highest
      in
      (bin_upper_bounds, offset+1)
  
  in
  (bin_upper_bounds, offset)
           


--find_bounds distinct_values counts num_distinct num_bins n min_data_bin
let find_bounds [l] (distinct_values: [l]f32) (counts: [l]i32) (num_bins: u16)
                    (total_num_samples: i64) (min_data_bin: i64) (zero_cnt: i64)
                    : []f32  =
  --let (neg_count) = partition (<= -zero_thres) distinct_values |> (.0) |> length
  -- let (neg_counts, neg_values) = map (<= -zero_thres) distinct_values |> zip counts
  --                          |> filter (.1) |> unzip
  let (negs, zero_plus) = zip distinct_values counts |> partition (\x -> x.0 <= -zero_thres)
  let (neg_values, neg_counts) = unzip negs
  --let split_i_neg = map i32.bool neg_counts |> i32.sum |> (\t -> t-1)
  let split_i_neg = length neg_values
  --let ha = trace (length neg_counts)
  let num_neg_samples = i32.sum neg_counts |> i64.i32
  let (zeros, pos) = partition (\x -> x.0 <= zero_thres) zero_plus
  let (pos_values, pos_counts) = unzip pos
  -- let (pos_counts, pos_b) = map (>= -zero_thres) distinct_values |> zip counts
  --                          |> filter (.1) |> unzip
  --let split_i_pos = map i32.bool pos_counts |> i32.sum |> (\t -> i32.i64 l-t-1) -- need off by one?
  let num_pos_samples = i32.sum pos_counts |> i64.i32
  let split_i_neg = if split_i_neg == 0 then l else split_i_neg
  --let ha = trace (split_i_neg, total_num_samples, num_bins-1)
  let left_max_bin =
    i64.f32 ((f32.i64 num_neg_samples)/(f32.i64 (total_num_samples-zero_cnt)) *f32.u16 (num_bins-1))
  let ha = trace left_max_bin
  let left_max_bin = i64.max 1 left_max_bin
  let hehe = trace (split_i_neg, num_bins)
  let (upper_bounds, neg_offset) =
    if (split_i_neg > 0) && (num_bins > 1) then
      -- bin upper bounds
      let ha = trace (length neg_counts, left_max_bin, num_neg_samples, min_data_bin)
      let (bin_upper_bounds, offset) =
        greedyFindBin neg_values neg_counts left_max_bin num_neg_samples min_data_bin
      in
        (bin_upper_bounds, offset)
    else
      (replicate left_max_bin 0.0f32, 0)
      
  --let split_i_pos = length pos_values - split_i_neg - ((unzip zeros).1 |> i32.sum |> i64.i32) -1
  let split_i_pos = length pos_values
  let ha = trace (upper_bounds, neg_offset)
  let upper_bounds = upper_bounds[:neg_offset]
  let right_max_bin = i64.u16 num_bins - 1 - neg_offset -- offset == length upper_bounds?!!
  let ha = trace(right_max_bin, split_i_pos)
  let (rest_upper_bounds, offset) = 
    if (split_i_pos >= 0) && (right_max_bin > 0) then
      let (new_bounds, offset) =
        greedyFindBin pos_values pos_counts right_max_bin num_pos_samples  min_data_bin
      in
         --(new_bounds with [offset] = zero_thres, offset+1)
         (new_bounds, offset)
    else
      -- add limit = infinity (f32.max)
      ([f32.highest], 1)

  let ha = trace (rest_upper_bounds, offset)
  --let final_bounds = upper_bounds ++ [-zero_thres] ++ rest_upper_bounds[:offset]
  in
  if length upper_bounds > 0 then
    if (split_i_pos >= 0) && (right_max_bin > 0) then
      upper_bounds[:neg_offset-1] ++ [-zero_thres] ++ [zero_thres] ++ rest_upper_bounds[:offset]
    else
      upper_bounds[:neg_offset-1] ++ [-zero_thres] ++ upper_bounds[:offset]
  else
    upper_bounds



let findBin [n] (vals: [n]f32) (num_bins: u16) : []f32 =
  if num_bins == 1u16 then
    [f32.maximum vals * 2]
  else
    let min_data_bin = 1
    let (na, rest) = partition (f32.isnan) vals
    let (zeros, rest) = partition (\x -> -zero_thres <= x && x <= zero_thres) rest
    let na_cnt = length na
    let zero_cnt = length zeros
    let num_samples = length rest
    let rest = rest :> [num_samples]f32
    let sorted = radix_sort_float f32.num_bits f32.get_bit rest
    -- unique for sorted values?.
    -- instead of != then abs(e-e1) < zero_thres for numircal stability?
    let unique_start = map2 (!=) sorted (rotate (-1) sorted)
    let distinct_values = zip sorted unique_start |> filter (.1) |> unzip |> (.0)
    let num = (length distinct_values)
    let distinct_values = distinct_values :> [num]f32
    let counts = segmented_reduce (+) 0 unique_start (replicate num_samples 1i32) num
    let (neg, pos) = zip distinct_values counts |> partition (\x -> x.0 < 0)
    let (neg_vals, neg_counts) = unzip neg
    let (pos_vals, pos_counts) = unzip pos
    let num = num +1
    let distinct_values = neg_vals ++ [0f32] ++ pos_vals :> [num]f32
    let counts = neg_counts ++ [i32.i64 zero_cnt] ++ pos_counts :> [num]i32
    -- let ha = trace distinct_values
    -- let ha = trace counts
    -- let ha = trace (length distinct_values, length counts)
    let upper_bounds =
      find_bounds distinct_values counts (num_bins-1u16) (n-na_cnt) min_data_bin zero_cnt
    let upper_bounds = upper_bounds ++ [f32.nan]
    in
    upper_bounds



let value_to_bin [n] (value: f32) (bin_bounds: [n]f32) (num_bins: u16) : u16 =
  let l = 0
  let r = num_bins -1
  let r = r-1
  in
  if f32.isnan value then
     num_bins-1
  else
    let (l, _) =
      loop (l, r) = (l, r) while l < r do
        let m = (r+l-1)/2
        in
          if i64.u16 m >= n || (value <= bin_bounds[i64.u16 m]) then
            (l, m)
          else
            (m+1, r)
    in
    l



let binMap [n] (vals: [n]f32) (num_bins: u16) : ([]u16, []f32) =
  let bounds = findBin vals num_bins
  let vals= radix_sort_float f32.num_bits f32.get_bit vals
  let mapped = map (\v -> value_to_bin v bounds num_bins) vals
  let numms = reduce_by_index (replicate (i64.u16 num_bins) 0u16) (+) 0u16 (map i64.u16 mapped) (replicate n 1u16)
  -- let numms = mapped
  in
    (numms, bounds)



let ha =
  let he = binMap woopdata[:,0] 10
  in
  (he.0, he.1, value_to_bin f32.nan he.1 10)
let main [n][d] (data: [n][d]f32) (labels: [n]f32) =
  let row = data[:, 0]
  let he = binMap row 10u16
  in
  he.1

-- let arr = [-10.0f32, 5.0, 3.0, -30.0, 2.0, 4.0, 0.0, 70.3, 12578.3, 3.2]
-- let num_bins = 30u16
-- let test = findBin arr num_bins
-- let res = map (\v -> value_to_bin v test num_bins) arr
    -- let unique = map2 (!=) sorted (rotate (-1) sorted) |> zip sorted |> filter (.1)
    -- let distinct_values = replicate (length unique+2) 0f32
    -- let counts = replicate (length unique+2) 0i32
    -- let (distinct_values, counts, offset) =
    --   if num_samples == 0 || (head sorted > 0f32 && zero_cnt > 0) then
    --     (distinct_values with [0] = 0f32, counts with [0] = i32.i64 zero_cnt, 1i64)
    --   else
    --     (distinct_values, counts, 0)
        
    -- let (distinct_values, counts, offset) =
    --   if num_samples > 0 then
    --     (distinct_values with [offset] = sorted[0], counts with [offset]=1, offset+1)
    --   else
    --     (distinct_values, counts, offset)
        
    -- let (distinct_values, counts, offset) =
    --   loop (distinct_values, counts, offset) =
    --     (distinct_values, counts, offset) for i < num_samples-1 do
    --       --let j = i+1
    --       let prev = sorted[i]
    --       let cur = sorted[i+1]
    --       in
    --         if (!(f32.abs(prev-cur) <= zero_thres)) then
    --           let (tmp_vals, tmp_counts, tmp_offset) = 
    --             if (prev < 0 && cur > 0) then
    --               (distinct_values with [offset+1]=0f32,
    --                counts with [offset+1] = i32.i64 zero_cnt, offset+1)
    --             else
    --               (distinct_values, counts, offset)
    --           in
    --           (tmp_vals with [tmp_offset+1]= cur, tmp_counts with [tmp_offset+1]=1, tmp_offset+1)
    --         else
    --           let num_distinct = offset
    --           let new_vals = distinct_values with [num_distinct] = cur
    --           let new_counts = counts with [num_distinct] = counts[num_distinct] + 1
    --           --let ha = trace (num_distinct, new_counts[num_distinct])
    --           --let ha = trace (new_counts[:offset])
    --           in
    --           (new_vals, new_counts, offset)
    -- --let ha = trace counts
    -- let (distinct_values, counts, offset) =
    --   if (num_samples > 0) && ((last sorted) < 0 && zero_cnt > 0) then
    --     (distinct_values with [offset] = 0f32, counts with [offset] = i32.i64 zero_cnt, offset+1)
    --   else
    --     (distinct_values, counts, offset)
