# Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization
Official implementation of paper:

> "Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization"
> Tiancheng Xing, Jerry Li, Yixuan Du, Xiyang Hu

<div align=center>
<img width=100% src="./img/raf.png"/>
</div>

## Our Improvement: Diversity-Regularized RAF

**Motivation.** Vanilla RAF optimizes each target product independently. As a result, adversarial prompts for semantically unrelated products often collapse into near-identical templates (mode collapse), which hurts stealth and realism.

**Method.** We add a semantic diversity penalty against a history of previously optimized prompts:

```
L_total  = w_ctrl_2 * L_read + w_tar_2 * L_tar + w_div * L_div
L_div(x) = max_j cos( E(x), E(h_j) )
```

where `E(.)` is a sentence encoder (`sentence-transformers/all-MiniLM-L6-v2`) and `h_j` are final prompts of previously attacked products. The penalty is computed **per token candidate at every stage-2 optimization step**, so collapsing candidates are pruned during the search rather than post-hoc.

**Backward compatible.** With `--w_div 0.0` (the default), no diversity module is loaded and the behavior is identical to the original RAF implementation.

## Environment for the improved version

The diversity module requires `sentence-transformers`. On our environment (Python 3.9, PyTorch 2.1.2+cu118), the following pinned versions are required — newer `sentence-transformers` upgrades `transformers` and breaks PyTorch 2.1.x:

```bash
pip install transformers==4.36.2 tokenizers==0.15.2 sentence-transformers==2.2.2 huggingface_hub==0.21.4
```

The MiniLM encoder (~80 MB) is downloaded automatically on first use. In mainland China you may want `export HF_ENDPOINT=https://hf-mirror.com`.

## Run

### Optimization
```bash
bash run_raf.sh
```
The default script runs the improved experiment: attacking products 4 and 6 of `coffee_machines` with `w_div=10`. Product 4 is optimized with an empty history; its final prompt is then added to the history, so product 6 is optimized under `L_div`.

To reproduce the original baseline (no diversity penalty):
```bash
bash run_raf.sh --w_div 0.0 --result_dir result/raf_main
```

### Evaluation

Ranking / perplexity evaluation of the optimized prompts:
```bash
bash eval_ranking.sh
```

Cross-product prompt similarity (mode-collapse measurement):
```bash
# single group
bash eval_similarity.sh --result_dir result/raf_pro_w10/deepseek-7b/json/coffee_machines --indices 4 6

# baseline vs improved
bash eval_similarity.sh --compare \
    --dir_a result/raf_main/deepseek-7b/json/coffee_machines \
    --dir_b result/raf_pro_w10/deepseek-7b/json/coffee_machines \
    --indices 4 6
```

## What changed in the code

| File | Description |
|---|---|
| `experiment/diversity.py` | New. `DiversityPenalty` (MiniLM encoding, prompt history bank, `L_div`), unified decoding and HTML-tag cleaning |
| `experiment/attack_raf.py` | `stage_two_search` computes `L_div` for every candidate at each step and returns it; combined loss includes `w_div * L_div` |
| `experiment/main.py` | New `--w_div` argument; multi-product loop (`--target_product_idx` accepts several indices); history update after each attack |
| `test_similarity.py` | New. Pairwise cosine-similarity evaluation across products (auto-detects `raf_results.csv` / `autodan_results.csv`) |
| `eval_similarity.sh` | New. One-command similarity evaluation / comparison |
| `eval_ranking.sh` | Renamed from `evaluate.sh` |

## Citation
If you find our code useful for your research, please cite our paper.
```bibtex
@article{xing2025llms,
  title={Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization},
  author={Xing, Tiancheng and Li, Jerry and Du, Yixuan and Hu, Xiyang},
  journal={arXiv preprint arXiv:2510.06732},
  year={2025}
}
```
