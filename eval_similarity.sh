#!/bin/bash
# 评估攻击 prompt 之间的跨样本语义相似度
# 用于衡量 mode collapse 程度
#
# 用法：
#   1. 评估单组实验（如原版 RAF 基线）：
#      bash eval_similarity.sh --result_dir result/raf_main/deepseek-7b/json/coffee_machines
#
#   2. 评估改进实验：
#      bash eval_similarity.sh --result_dir result/raf_pro_w10/deepseek-7b/json/coffee_machines
#
#   3. 对比两组实验：
#      bash eval_similarity.sh --compare \
#          --dir_a result/raf_main/deepseek-7b/json/coffee_machines \
#          --dir_b result/raf_pro_w10/deepseek-7b/json/coffee_machines

# 默认参数（基线实验）
RESULT_DIR="result/raf_main/deepseek-7b/json/coffee_machines"
INDICES="1 2 3 4 5 6 7 8 9 10"

# 运行评估
python test_similarity.py \
    --result_dir "$RESULT_DIR" \
    --indices $INDICES \
    "$@"
