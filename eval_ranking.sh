#!/bin/bash
python test_multiple_ranking.py \
    --csv_file_template "result/raf_pro_w10/{model}/json/{catalog}/{target_idx}/autodan_results.csv" \
    --model deepseek-7b \
    --catalog "coffee_machines" \
    --target_product_idx 4 6 \
    --dataset json \
    --num_runs 10 \
    --random_order