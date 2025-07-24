# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


@torch.fx.wrap
def _get_valid_attn_mask(
    device: torch.device,           # 计算设备
    causal: bool,                   # 是否启用因果掩码
    N: int,                         # 填充后的序列长度
    seq_lengths: torch.Tensor,      # 实际序列长度
    num_targets: Optional[torch.Tensor] = None,  # 目标数量（推荐特有）
    max_attn_len: int = 0,          # 最大注意力长度限制
    contextual_seq_len: int = 0,    # 上下文序列长度
    min_full_attn_seq_len: int = 0, # 最小完整注意力长度
) -> torch.Tensor:
    ids = torch.arange(0, N, device=device).view(1, N)  # [1, N]: [0,1,2,...,N-1]
    max_ids = seq_lengths.view(-1, 1, 1)                # [B, 1, 1]: 每个序列的实际长度
    # 序列结构: [上下文特征] + [用户历史] + [候选物品]
    # 长度:     [contextual] + [user_history] + [candidates]
    # contextual_seq_len = 2 表示前2个位置是上下文
    # 调整后的ids让用户历史从位置0开始计算
    if contextual_seq_len > 0:
        ids = ids - contextual_seq_len + 1     # 调整位置索引
        ids = torch.clamp(ids, min=0)          # 确保非负
        max_ids = max_ids - contextual_seq_len + 1  # 调整有效长度
    if num_targets is not None:     # 此处实现target目标无法看到彼此，只能看到用户历史
        max_ids = max_ids - num_targets.view(-1, 1, 1)  # 排除候选目标
        ids = torch.clamp(ids, max=max_ids)              # 限制有效范围
        
        # 构建2D位置矩阵
        row_ids = ids.view(-1, N, 1).expand(-1, N, N)   # 行位置
        col_ids = ids.view(-1, 1, N).expand(-1, N, N)   # 列位置
    else:
        # 标准注意力：所有位置可见
        row_ids = ids.view(N, 1).expand(N, N)
        col_ids = row_ids.t()
        row_ids = row_ids.view(1, N, N)
        col_ids = col_ids.view(1, N, N)
    row_col_dist = row_ids - col_ids
    valid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
    if not causal:
        # 非因果：双向注意力，使用绝对距离
        row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)
    # 构建基础有效掩码
    valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)
    if max_attn_len > 0:
        if min_full_attn_seq_len > 0:
            # 复杂策略：长序列全注意力，短序列限制注意力。序列末尾的物品（最近交互）对推荐更重要，候选物品可以访问完整的用户历史
            # 序列前段mask部分控制复杂度，序列末段不mask确保质量
            valid_attn_mask = torch.logical_and(
                valid_attn_mask,
                torch.logical_or(
                    row_col_dist <= max_attn_len,           # 距离限制, > max_attn_len 的都屏蔽
                    row_ids >= max_ids - min_full_attn_seq_len,  # 末尾部分，获得完整注意力
                ),
            )
        else:
            # 简单策略：全局距离限制，距离 > max_attn_len 的都屏蔽
            valid_attn_mask = torch.logical_and(
                valid_attn_mask, row_col_dist <= max_attn_len
            )
    if contextual_seq_len > 0:
        # 上下文位置可以注意到所有有效位置
        valid_attn_mask = torch.logical_or(
            valid_attn_mask, torch.logical_and(row_ids == 0, col_ids < max_ids)
        )
    return valid_attn_mask


def _pad_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L, H, D = q.shape
    V = v.shape[2]
    padded_q = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=q.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]
    padded_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]
    padded_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(L, H * V),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, V)
        .transpose(1, 2)
    )  # [B, H, N, D]
    return padded_q, padded_k, padded_v


@torch.fx.wrap
def pytorch_hstu_mha(
    max_seq_len: int,   # 最大序列长度
    alpha: float,       # 缩放因子，通常为 \frac{1}{\sqrt{d_k}}
    q: torch.Tensor,    # 查询张量 [L, H, D]
    k: torch.Tensor,    # 键 [L, H, D]
    v: torch.Tensor,    # 值 [L, H, V]
    seq_offsets: torch.Tensor, # 偏移量
    causal: bool = True,    # 是否使用因果掩码
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None, # 目标数量
    max_attn_len: int = 0,   # 最大注意力长度
    contextual_seq_len: int = 0, # 上下文长度
    min_full_attn_seq_len: int = 0, # 最小完整注意力长度
) -> torch.Tensor:
    L, H, _ = q.shape   # L=总长度, H=头数, _=注意力维度
    V = v.shape[2]  # V=值的维度
    q, k, v = _pad_qkv(
        q, k, v, seq_offsets, max_seq_len
    )  # [B, H, N, D) and [B, H, N, V]
    qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
    qk_attn = F.silu(qk_attn) / max_seq_len
    valid_attn_mask = _get_valid_attn_mask(
        device=q.device,
        causal=causal,              # 因果掩码
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1], # 实际序列长度
        num_targets=num_targets,    # 推荐目标数量
        max_attn_len=max_attn_len,   # 最大注意力长度   
        contextual_seq_len=contextual_seq_len, # 上下文长度
        min_full_attn_seq_len=min_full_attn_seq_len,
    )
    # raise NotImplementedError(valid_attn_mask[0, :, :].to(torch.int32))
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    if dropout_pr > 0.0:
        qk_attn = F.dropout(qk_attn, p=dropout_pr, training=training)
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)  # [B, H, N, V]
    return torch.ops.fbgemm.dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),  # [B, H, N, V]->[B, N, H, V]->->[B, N, H * V]
        [seq_offsets],
        L,
    )[0].view(L, H, V)


@torch.fx.wrap
def pytorch_cached_hstu_mha(
    max_seq_len: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    L, H, D = delta_q.shape
    _, _, V = v.shape
    B = seq_offsets.size(0) - 1
    delta_size = L // B
    delta_q = delta_q.view(B, -1, H, D).transpose(1, 2) # [L, H, D] -> [B, L, H, D] -> [B, H, L, D]
    full_k = (  # K: [N, H, D] -> [B, max_seq_len, H, D] -> [B, H, max_seq_len, D]
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(-1, H * D),
            offsets=[seq_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )
        .view(B, -1, H, D)
        .transpose(1, 2)
    )
    full_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(-1, H * V),
            offsets=[seq_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )
        .view(B, -1, H, V)
        .transpose(1, 2)
    )
    qk_attn = torch.einsum("bhxa,bhya->bhxy", delta_q, full_k) * alpha  #[B, H, delta_size, D] × [B, H, max_seq_len, D] -> [B, H, delta_size, max_seq_len]
    qk_attn = F.silu(qk_attn) / max_seq_len
    full_valid_attn_mask = _get_valid_attn_mask(
        device=delta_q.device,
        causal=True,
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    mask = torch.arange(max_seq_len, device=delta_q.device).view(1, -1)
    # 生成掩码，只关注新增token的有效注意力范围。最终掩码 [B, max_seq_len]，用户1: [19-38]有效，用户2: [24-43]有效
    mask = torch.logical_and(
        mask >= (seq_lengths - delta_size).view(-1, 1), # 从新增token开始位置
        mask < seq_lengths.view(-1, 1), # 到当前序列结束位置
    )
    valid_attn_mask = (
        full_valid_attn_mask.expand(B, -1, -1) # [max_seq_len, max_seq_len] -> [B, max_seq_len, max_seq_len] -> [B*max_seq_len, max_seq_len]
        .flatten(0, 1)[mask.view(-1), :]    # 只保留新增token行
        .view(-1, delta_size, max_seq_len)  # [B, delta, max_seq_len]
    )
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    attn_output = torch.einsum("bhxd,bhdv->bhxv", qk_attn, full_v)
    return attn_output.transpose(1, 2).reshape(-1, H, V)
