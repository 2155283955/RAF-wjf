"""
多样性惩罚模块（Diversity Regularization）

目的：缓解 RAF 攻击中的 mode collapse 问题 —— 多个不同产品收敛到高度相似的攻击短语。
方法：对每个候选攻击 prompt，计算其与历史已选 prompt 的最大余弦相似度作为惩罚项 L_div。

使用 sentence-transformers/all-MiniLM-L6-v2（384 维句向量）计算语义相似度。
与 DeepSeek-7B 的语义空间存在一定偏差，但 MiniLM 专为语义相似度优化，
足以衡量"攻击 prompt 文本是否过于相似"。
"""

import torch
import torch.nn.functional as F
from typing import List
from sentence_transformers import SentenceTransformer


def decode_for_diversity(tokenizer, token_ids):
    """
    统一解码函数：compute_div_loss 和 add_to_history 共用，
    确保送入句向量模型的文本与存入历史的文本完全一致，
    避免编码/解码不一致带来的文本差异导致相似度计算不准。

    Args:
        tokenizer: 主模型的 tokenizer
        token_ids: token id 序列（tensor 或 list）

    Returns:
        str: 纯文本（不含 special tokens）
    """
    # 如果传入的是 tensor，先转为 list
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def clean_span_tags(text: str) -> str:
    """
    清洗 csv 中 attack_prompt 字段携带的 HTML span 标签。
    训练时加入历史库的是纯文本，评估脚本读取 csv 后必须做同样的清洗，
    否则相似度计算会被 HTML 标签污染。

    注意：不能用 lstrip/rstrip，它们是字符集匹配，会过度清洗。
    必须用 replace 替换整个标签字符串。

    例：'<span style="color:red;"> This Espresso machine</span>' → 'This Espresso machine'
    """
    text = text.replace('<span style="color:red;">', '')
    text = text.replace('</span>', '')
    return text.strip()


class DiversityPenalty:
    """
    基于句向量余弦相似度的多样性惩罚模块。
    使用 sentence-transformers/all-MiniLM-L6-v2（384 维句向量），
    与 DeepSeek-7B 语义空间存在一定偏差，但足以衡量"攻击 prompt 是否过于相似"。

    总损失公式：
        L_total = w_ctrl_2 * L_read + w_tar_2 * L_tar + w_div * L_div
        L_div(x'_i) = max_j cos(encode(x'_i), history_j)
    """

    def __init__(self, device='cuda'):
        """
        初始化多样性惩罚模块。

        Args:
            device: 设备标识，必须和主模型保持一致（'cuda' 或 'cpu'）
        """
        self.device = device

        # 加载轻量级句向量模型（~80MB），与主模型（~14GB）独立，显存开销极小
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        # 动态获取输出维度（MiniLM-L6-v2 固定 384 维，但用代码获取更稳健）
        dummy_emb = self.model.encode(["dummy"], convert_to_tensor=True)
        embed_dim = dummy_emb.shape[-1]

        # 初始化空历史库，设备和主模型一致，dtype 为 float32
        self.history_embeddings = torch.empty(
            (0, embed_dim), dtype=torch.float32, device=self.device
        )
        self.history_prompts: List[str] = []

        print(f"[Diversity] DiversityPenalty initialized on {self.device}, "
              f"embed_dim={embed_dim}")

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        批量编码文本为句向量。
        使用 convert_to_tensor=True + 强制 to(device, dtype)，
        防止 sentence-transformers 某些版本异常返回 cpu tensor。

        Args:
            texts: 文本列表

        Returns:
            Tensor, shape=(len(texts), embed_dim), dtype=float32, device=self.device
        """
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        # 强制确保在正确设备上、统一 dtype
        return embeddings.to(device=self.device, dtype=torch.float32)

    def compute_div_loss(self, candidate_texts: List[str]) -> torch.Tensor:
        """
        计算 L_div 多样性惩罚损失。

        Args:
            candidate_texts: List[str], 长度 = topk（候选数量）
                每个元素是当前候选 token 拼接 fixed_control_sequence 后的
                临时攻击序列文本（用 decode_for_diversity 统一解码）。

        Returns:
            div_losses: Tensor, shape = (topk,), dtype = float32, device = self.device
                L_div(x'_i) = max_j cos(encode(x'_i), history_j)
                即当前候选与所有历史 prompt 的最大余弦相似度。
                值域 [0, 1]：0 表示与历史完全不相似，1 表示完全相同。
        """
        num_candidates = len(candidate_texts)

        # 空历史库保护：第一个产品攻击时没有历史，返回全 0
        # shape 和候选数量严格匹配，防止后续 matmul 维度报错
        if self.history_embeddings.shape[0] == 0:
            return torch.zeros(
                num_candidates, dtype=torch.float32, device=self.device
            )

        # 编码当前候选 → [topk, embed_dim]
        candidate_embs = self.encode(candidate_texts)

        # 计算余弦相似度矩阵: [topk, history_size]
        # candidate_embs.unsqueeze(1)  → [topk, 1, embed_dim]
        # history_embeddings.unsqueeze(0) → [1, history_size, embed_dim]
        similarity_matrix = F.cosine_similarity(
            candidate_embs.unsqueeze(1),
            self.history_embeddings.unsqueeze(0),
            dim=2
        )

        # 取每行最大值：每个候选与历史中最相似的那个的相似度
        div_losses = similarity_matrix.max(dim=1).values  # [topk]

        # 确保输出在正确设备上、dtype 一致
        return div_losses.to(device=self.device, dtype=torch.float32)

    def add_to_history(self, prompt_text: str):
        """
        将选中的 attack prompt 加入历史记录。
        新 embedding 和旧 history_embeddings 全部放在 self.device 上，
        不会出现一部分 cpu 一部分 cuda 的情况。

        Args:
            prompt_text: 纯文本（已用 decode_for_diversity 解码）
        """
        new_emb = self.encode([prompt_text])  # [1, embed_dim]

        # 拼接后确保设备一致
        self.history_embeddings = torch.cat([
            self.history_embeddings,
            new_emb
        ], dim=0).to(device=self.device, dtype=torch.float32)

        self.history_prompts.append(prompt_text)

    def get_history_size(self) -> int:
        """返回历史 prompt 数量"""
        return len(self.history_prompts)
