"""
跨样本相似度评估脚本

评估多个产品的 attack prompt 之间的语义相似度，用于衡量 mode collapse 程度。
对比 w_div=0（基线 RAF）vs w_div=50（改进 RAF + L_div）的多样性差异。

关键：csv 中的 attack_prompt 字段带有 <span style="color:red;"> 标签，
必须用 clean_span_tags() 清洗后再编码，否则 HTML 标签会污染相似度计算。

用法：
    python similarity_eval.py \
        --result_dir result/raf_pro/deepseek-7b/json/coffee_machines \
        --indices 1 2 3 4 5 6 7 8 9 10

    # 对比两组实验
    python similarity_eval.py \
        --compare \
        --dir_a result/raf/deepseek-7b/json/coffee_machines \
        --dir_b result/raf_pro/deepseek-7b/json/coffee_machines \
        --indices 1 2 3 4 5 6 7 8 9 10
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tabulate import tabulate

from experiment.diversity import DiversityPenalty, clean_span_tags


def load_attack_prompts(result_dir, indices):
    """
    从多个产品目录中加载最佳 attack prompt。

    对每个产品目录，读取 autodan_results.csv，取 product_rank 最小（即排名最高）
    的那一行作为该产品的最佳 attack prompt。

    关键：csv 中的 attack_prompt 字段带有 <span style="color:red;"> 标签，
    必须用 clean_span_tags() 清洗，否则 HTML 标签会污染相似度计算。

    Args:
        result_dir: 结果根目录，如 result/raf_pro/deepseek-7b/json/coffee_machines
        indices: 产品索引列表

    Returns:
        prompts: List[str], 清洗后的纯文本 prompt 列表
        product_names: List[str], 对应的产品名
        raw_prompts: List[str], 原始未清洗的 prompt（用于展示）
    """
    prompts = []
    product_names = []
    raw_prompts = []

    for idx in indices:
        # 兼容两种文件名：论文实验用 raf_results.csv，新实验用 autodan_results.csv
        csv_path = os.path.join(result_dir, str(idx), 'raf_results.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(result_dir, str(idx), 'autodan_results.csv')
        if not os.path.exists(csv_path):
            print(f"  [Warning] File not found for index {idx}, skipping")
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"  [Warning] Empty CSV: {csv_path}, skipping")
            continue

        # 取最后一行作为最终优化的 prompt（最后一行是最终收敛结果）
        best_row = df.iloc[-1]

        raw_prompt = best_row['attack_prompt']
        # 清洗 HTML span 标签，确保与训练时加入历史库的纯文本一致
        clean_prompt = clean_span_tags(raw_prompt)

        prompts.append(clean_prompt)
        raw_prompts.append(raw_prompt)
        product_names.append(f"product_{idx}")

    return prompts, product_names, raw_prompts


def compute_similarity_matrix(prompts, device='cuda'):
    """
    计算所有 prompt 之间的 pairwise 余弦相似度矩阵。

    Args:
        prompts: List[str], 纯文本 prompt 列表
        device: 计算设备

    Returns:
        sim_matrix: np.ndarray, shape=(n, n), 余弦相似度矩阵
        embeddings: torch.Tensor, shape=(n, embed_dim)
    """
    # 使用 DiversityPenalty 中的 MiniLM 模型编码
    dp = DiversityPenalty(device=device)

    # 批量编码所有 prompt
    embeddings = dp.encode(prompts)  # [n, embed_dim]

    # 计算 pairwise 余弦相似度
    sim_matrix = F.cosine_similarity(
        embeddings.unsqueeze(0),  # [1, n, embed_dim]
        embeddings.unsqueeze(1),  # [n, 1, embed_dim]
        dim=2
    )  # [n, n]

    sim_matrix = sim_matrix.cpu().numpy()

    # 清理 MiniLM 模型显存
    del dp
    torch.cuda.empty_cache()

    return sim_matrix, embeddings


def evaluate_single(result_dir, indices, device='cuda'):
    """
    评估单组实验的跨样本相似度。

    Args:
        result_dir: 结果根目录
        indices: 产品索引列表
        device: 计算设备

    Returns:
        dict: 包含统计指标的字典
    """
    print(f"\nLoading prompts from: {result_dir}")
    prompts, product_names, raw_prompts = load_attack_prompts(result_dir, indices)

    if len(prompts) < 2:
        print(f"  [Error] Need at least 2 prompts, got {len(prompts)}")
        return None

    print(f"  Loaded {len(prompts)} prompts")
    print(f"\n  Clean prompts (for encoding):")
    for i, (name, prompt) in enumerate(zip(product_names, prompts)):
        print(f"    {name}: '{prompt}'")

    # 计算相似度矩阵
    sim_matrix, _ = compute_similarity_matrix(prompts, device)

    # 统计指标（排除对角线，即自身相似度 = 1.0）
    n = len(prompts)
    mask = ~np.eye(n, dtype=bool)
    upper_tri_values = sim_matrix[mask]  # 所有非对角线元素

    avg_sim = np.mean(upper_tri_values)
    max_sim = np.max(upper_tri_values)
    min_sim = np.min(upper_tri_values)
    std_sim = np.std(upper_tri_values)

    # 找到最相似的一对
    max_i, max_j = np.unravel_index(np.argmax(sim_matrix * (1 - np.eye(n))), sim_matrix.shape)

    print(f"\n  Similarity Statistics:")
    print(f"    Average: {avg_sim:.4f} ± {std_sim:.4f}")
    print(f"    Max:     {max_sim:.4f}  ({product_names[max_i]} vs {product_names[max_j]})")
    print(f"    Min:     {min_sim:.4f}")

    # 打印相似度矩阵
    print(f"\n  Pairwise Similarity Matrix:")
    header = [""] + product_names
    rows = []
    for i in range(n):
        row = [product_names[i]] + [f"{sim_matrix[i][j]:.3f}" for j in range(n)]
        rows.append(row)
    print(tabulate(rows, headers=header, tablefmt='grid'))

    return {
        'avg_sim': avg_sim,
        'max_sim': max_sim,
        'min_sim': min_sim,
        'std_sim': std_sim,
        'sim_matrix': sim_matrix,
        'prompts': prompts,
        'raw_prompts': raw_prompts,
        'product_names': product_names,
        'max_pair': (product_names[max_i], product_names[max_j]),
    }


def compare_two(result_dir_a, result_dir_b, indices, device='cuda'):
    """
    对比两组实验的跨样本相似度。

    Args:
        result_dir_a: 基线实验目录（如 w_div=0）
        result_dir_b: 改进实验目录（如 w_div=50）
        indices: 产品索引列表
        device: 计算设备
    """
    print(f"\n{'='*80}")
    print(f"COMPARING TWO EXPERIMENTS")
    print(f"  A (baseline):  {result_dir_a}")
    print(f"  B (improved):  {result_dir_b}")
    print(f"{'='*80}")

    result_a = evaluate_single(result_dir_a, indices, device)
    result_b = evaluate_single(result_dir_b, indices, device)

    if result_a is None or result_b is None:
        print("  [Error] One or both evaluations failed")
        return

    # 对比表格
    comparison = [
        ["Average Similarity", f"{result_a['avg_sim']:.4f}", f"{result_b['avg_sim']:.4f}",
         f"{result_b['avg_sim'] - result_a['avg_sim']:+.4f}"],
        ["Max Similarity", f"{result_a['max_sim']:.4f}", f"{result_b['max_sim']:.4f}",
         f"{result_b['max_sim'] - result_a['max_sim']:+.4f}"],
        ["Min Similarity", f"{result_a['min_sim']:.4f}", f"{result_b['min_sim']:.4f}",
         f"{result_b['min_sim'] - result_a['min_sim']:+.4f}"],
        ["Std", f"{result_a['std_sim']:.4f}", f"{result_b['std_sim']:.4f}",
         f"{result_b['std_sim'] - result_a['std_sim']:+.4f}"],
    ]

    print(f"\n{'='*80}")
    print(f"COMPARISON TABLE")
    print(f"{'='*80}")
    print(tabulate(
        comparison,
        headers=["Metric", "A (baseline)", "B (improved)", "Diff (B-A)"],
        tablefmt='grid'
    ))

    # 判断改进效果
    if result_b['avg_sim'] < result_a['avg_sim']:
        print(f"\n  ✅ 改进有效：平均相似度下降 {result_a['avg_sim'] - result_b['avg_sim']:.4f}")
    else:
        print(f"\n  ⚠️ 改进无效：平均相似度未下降")


def main():
    parser = argparse.ArgumentParser(description="评估跨样本 attack prompt 相似度")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="单组实验的结果目录")
    parser.add_argument("--indices", type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help="产品索引列表")
    parser.add_argument("--compare", action='store_true',
                        help="对比两组实验")
    parser.add_argument("--dir_a", type=str, default=None,
                        help="基线实验目录（对比模式）")
    parser.add_argument("--dir_b", type=str, default=None,
                        help="改进实验目录（对比模式）")
    parser.add_argument("--device", type=str, default='cuda',
                        help="计算设备")
    parser.add_argument("--output", type=str, default=None,
                        help="输出相似度矩阵到 CSV 文件")

    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    if args.compare:
        if args.dir_a is None or args.dir_b is None:
            print("Error: --compare mode requires --dir_a and --dir_b")
            return
        compare_two(args.dir_a, args.dir_b, args.indices, device)
    elif args.result_dir is not None:
        result = evaluate_single(args.result_dir, args.indices, device)
        if result is not None and args.output:
            # 保存相似度矩阵到 CSV
            df = pd.DataFrame(
                result['sim_matrix'],
                index=result['product_names'],
                columns=result['product_names']
            )
            df.to_csv(args.output)
            print(f"\nSimilarity matrix saved to {args.output}")
    else:
        print("Error: specify --result_dir or --compare with --dir_a and --dir_b")


if __name__ == "__main__":
    main()
