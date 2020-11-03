import "../lib/github.com/diku-dk/sorts/radix_sort"
import "../lib/github.com/diku-dk/segmented/segmented"
import "../util"

let zero_thres = 0.000000000001f32

let greedyFindBin [l] (distinct_values: [l]f32) (counts: [l]i32)
                      (max_bin: u16) (total_num_samples: i64) (min_data_bin: i64) =
  let bin_upper_bounds = replicate (i64.u16 max_bin) 0f32
  let offset = 0
  let (bin_upper_bounds, offset) =
    if (u16.i64 l <= max_bin) then
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
      (new_bounds with [new_offset]= f32.highest, offset+1)
    else
      let max_bin = if min_data_bin > 0 then
                    let max_bin = i64.max (i64.u16 max_bin) total_num_samples/(i64.u16 max_bin)
                    in
                      i64.max max_bin 1
                    else
                      i64.u16 max_bin
      let mean_bin_size = f32.i64 total_num_samples / f32.i64 max_bin
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
      let is_big_count = is_big_count[:big_offset]
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
            let cur_in_bin = cur_in_bin + counts[i]
            let (upper_bounds, bin_cnt, lower_bounds, cur_in_bin, mean_bin_size, rest_bin_cnt) =
              if is_big_count[i] || (f32.i32 cur_in_bin >= mean_bin_size) ||
                 (is_big_count[i+1] && f32.i32 cur_in_bin >= f32.max 1f32 (mean_bin_size*0.5)) then
                let upper_bounds = upper_bounds with [bin_cnt] = distinct_values[i]
                let bin_cnt = bin_cnt+1
                let lower_bounds = lower_bounds with [bin_cnt] = distinct_values[i+1]
                -- implement some sort of break here! maybe if and then if ?
                let cur_in_bin = 0
                let (rest_bin_cnt, mean_bin_size) =
                  if !is_big_count[i] then
                    (rest_bin_cnt-1, f32.i64 rest_sample_cnt / f32.i64 rest_bin_cnt)
                  else
                    (rest_bin_cnt, mean_bin_size)
                in
                  (upper_bounds, bin_cnt, lower_bounds, cur_in_bin, mean_bin_size, rest_bin_cnt)
              else
                (upper_bounds, bin_cnt, lower_bounds, cur_in_bin, mean_bin_size, rest_bin_cnt)
            in
            (upper_bounds, lower_bounds, bin_cnt, rest_sample_cnt,
             cur_in_bin, mean_bin_size, rest_bin_cnt)
      let bin_cnt = bin_cnt +1
      let (bin_upper_bounds, offset) =
        loop (bin_upper_bounds, offset) = (bin_upper_bounds, 0) for i < bin_cnt - 1 do
          let value = (upper_bounds[i] + lower_bounds[i+1])/2f32
          in
            if (offset == 0) || !(bin_upper_bounds[offset] >= value) then -- maybe just >?
              (bin_upper_bounds with [offset] = value, offset+1)
            else
              (bin_upper_bounds, offset)
      let bin_upper_bounds = bin_upper_bounds with [offset] = f32.highest
      in
      (bin_upper_bounds, offset+1)
  
  in
  bin_upper_bounds[:offset]
           


--find_bounds distinct_values counts num_distinct num_bins n min_data_bin
let find_bounds [l] (distinct_values: [l]f32) (counts: [l]i32) (num_bins: u16)
                    (total_num_samples: i64) (min_data_bin: i64) (zero_cnt: i64) =
  --let (neg_count) = partition (<= -zero_thres) distinct_values |> (.0) |> length
  let (neg_counts, neg_b) = map (<= -zero_thres) distinct_values |> zip counts
                           |> filter (.1) |> unzip
  let split_i_neg = map i32.bool neg_b |> i32.sum |> (\t -> t-1)
  let num_neg_samples = i32.sum neg_counts
  --let pos_count = partition (=>zero_thres) distinct_values |> (.0) |> length
  let (pos_counts, pos_b) = map (>= -zero_thres) distinct_values |> zip counts
                           |> filter (.1) |> unzip
  let split_i_pos = map i32.bool pos_b |> i32.sum |> (\t -> i32.i64 l-t-1) -- need off by one?
  let num_pos_samples = i32.sum pos_counts
  let split_i_neg = if split_i_neg < 0 then i32.i64 l else split_i_neg
  let haha =
    if (split_i_neg > 0) && (num_bins > 1) then
      let left_max_bin =
        i32.f32 ((f32.i32 split_i_neg)/(f32.i64 (l-zero_cnt)) *f32.u16 (num_bins-1))
      let left_max_bin = i32.max 1 left_max_bin
      -- bin upper bounds
      -- if (length bin_upper_bounds) > 0
      -- then [length bin_upper_bounds-1] = -zero bound
      -- else bin_upper_bounds
      in
        left_max_bin
    else
      0
  --let right_max_bin = i32.u16 num_bins -1 - i32.i64 offset?
  --let hehe =
    --if (split_i_pos >= 0) && (right_max_bin > 0) then
    -- distinct_vals[split_i_pos:] counts[split_i_pos:] l-split_i_pos-1? right_max_bin right_cnt_data
    -- call find binds

     -- append zero bound to bin_upper_bounds
     -- add found bounds right
     -- else
     -- add limit = infinity (f32.max)
  -- tdb
  in
     --distinct_values
     replicate (i64.i32 haha) 0f32



let findBin [n] (vals: [n]f32) (num_bins: u16) =
  let min_data_bin = 1
  let (na, rest) = partition (==f32.nan) vals
  let (zeros, rest) = partition (\x -> -zero_thres <= x && x <= zero_thres) rest
  let na_cnt = length na
  let zero_cnt = length zeros
  let num_samples = length rest
  let sorted = radix_sort_float f32.num_bits f32.get_bit rest
  -- unique for sorted values?.
  -- instead of != then abs(e-e1) < zero_thres for numircal stability?
  let unique = map2 (!=) sorted (rotate (-1) sorted) |> zip sorted |> filter (.1)
  let distinct_values = replicate (length sorted+2) 0f32
  let counts = replicate (length sorted+2) 0i32
  let (distinct_values, counts, offset) =
    if num_samples == 0 || (head sorted > 0f32 && zero_cnt > 0) then
      (distinct_values with [0] = 0f32, counts with [0] = i32.i64 zero_cnt, 1i64)
    else
      (distinct_values, counts, 0)

  let (distinct_values, counts, offset) =
    if num_samples > 0 then
      (distinct_values with [offset] = sorted[0], counts with [offset]=1, offset+1)
    else
      (distinct_values, counts, offset)
  let (distinct_values, counts, offset) =
    loop (distinct_values, counts, offset) =
      (distinct_values, counts, offset) for i < num_samples-1 do
        let j = i+1
        let prev = sorted[j-1]
        let cur = sorted[j]
        in
          if (!(f32.abs(prev-cur) <= zero_thres)) then
            let (tmp_vals, tmp_counts, tmp_offset) = 
              if (prev < -zero_thres && cur < zero_thres) then
                (distinct_values with [offset]=0f32,
                 counts with [offset] = i32.i64 zero_cnt, offset+1)
              else
                (distinct_values, counts, offset)
          in
            (tmp_vals with [tmp_offset]= cur, tmp_counts with [tmp_offset]=1, tmp_offset+1)
        else
          let num_distinct = offset
          let new_vals = distinct_values with [num_distinct] = cur
          let new_counts = counts with [num_distinct] = last counts + 1
          in
          (new_vals, new_counts, offset+1)
  let (distinct_values, counts, offset) =
    if (num_samples > 0) && ((last sorted) < -zero_thres && zero_cnt > 0) then
      (distinct_values with [offset] = 0f32, counts with [offset] = i32.i64 zero_cnt, offset+1)
    else
      (distinct_values, counts, offset)
  let distinct_values = distinct_values[:offset]
  let counts = counts[:offset]
  let min_val = head distinct_values
  let max_val = last distinct_values
  let num_distinct = length distinct_values
  let upper_bounds =
    find_bounds distinct_values counts num_bins n min_data_bin zero_cnt
                
  let upper_bounds = upper_bounds ++ [f32.nan]
  in
  upper_bounds
