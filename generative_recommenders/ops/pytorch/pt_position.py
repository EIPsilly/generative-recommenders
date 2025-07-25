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

from typing import Optional

import torch
from generative_recommenders.common import (
    fx_unwrap_optional_tensor,
    jagged_to_padded_dense,
)

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


@torch.fx.wrap
def torch_arange(end: int, device: torch.device) -> torch.Tensor:
    return torch.arange(end, device=device)


@torch.fx.wrap
def _get_col_indices(
    max_seq_len: int,
    max_contextual_seq_len: int,
    max_pos_ind: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
) -> torch.Tensor:
    # 例如
    # max_seq_len = 10              # 最大序列长度
    # max_contextual_seq_len = 2    # 上下文特征数量
    # num_targets = [2, 3]          # 每个用户的候选物品数
    # seq_lengths = [8, 10]         # 每个用户的实际序列长度
    # max_pos_ind = 8192           # 位置嵌入表大小
    # B = 2                        # 用户数量
    # col_indices = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    B = seq_lengths.size(0)
    col_indices = torch.arange(max_seq_len, device=seq_lengths.device).expand(
        B, max_seq_len
    )
    # 处理目标位置（如果有候选物品）
    # 计算有效历史长度（总长度 - 候选数量）
    # high_inds = seq_lengths - num_targets = [8-2, 10-3] = [6, 7]
    # # 将索引限制在有效历史范围内
    # col_indices = torch.clamp(col_indices, max=high_inds.view(-1, 1))
    # high_inds.view(-1, 1) = [[6], [7]]
    # clamp后：
    # col_indices = [[0, 1, 2, 3, 4, 5, 6, 6, 6, 6],  # 用户1：超过6的都变成6
    #                [0, 1, 2, 3, 4, 5, 6, 7, 7, 7]]  # 用户2：超过7的都变成7
    # 相对位置 = 历史结束位置 - 当前位置
    # col_indices = [[6, 5, 4, 3, 2, 1, 0, 0, 0, 0],
    #                [7, 6, 5, 4, 3, 2, 1, 0, 0, 0]]
    if num_targets is not None:
        if interleave_targets:
            high_inds = seq_lengths - fx_unwrap_optional_tensor(num_targets) * 2
        else:
            high_inds = seq_lengths - fx_unwrap_optional_tensor(num_targets)
        col_indices = torch.clamp(col_indices, max=high_inds.view(-1, 1))
        col_indices = high_inds.view(-1, 1) - col_indices
    else:
        col_indices = seq_lengths.view(-1, 1) - col_indices
    # 添加上下文偏移(+2)
    # col_indices = [[8, 7, 6, 5, 4, 3, 2, 2, 2, 2],
    #                [9, 8, 7, 6, 5, 4, 3, 2, 2, 2]]
    col_indices = col_indices + max_contextual_seq_len
    col_indices = torch.clamp(col_indices, max=max_pos_ind - 1)
    # 前max_contextual_seq_len个位置设为固定值[0, 1]
    # 最终结果：
    # col_indices = [[0, 1, 6, 5, 4, 3, 2, 2, 2, 2],
    #                [0, 1, 7, 6, 5, 4, 3, 2, 2, 2]]
    if max_contextual_seq_len > 0:
        col_indices[:, :max_contextual_seq_len] = torch.arange(
            0,
            max_contextual_seq_len,
            device=col_indices.device,
            dtype=col_indices.dtype,
        ).view(1, -1)
    return col_indices


def pytorch_add_timestamp_positional_embeddings(
    seq_embeddings: torch.Tensor,
    seq_offsets: torch.Tensor,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    max_seq_len: int,
    max_contextual_seq_len: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
) -> torch.Tensor:
    max_pos_ind = pos_embeddings.size(0)
    # position encoding
    pos_inds = _get_col_indices(
        max_seq_len=max_seq_len,
        max_contextual_seq_len=max_contextual_seq_len,
        max_pos_ind=max_pos_ind,
        seq_lengths=seq_lengths,
        num_targets=num_targets,
        interleave_targets=interleave_targets,
    )
    B, _ = pos_inds.shape
    # timestamp encoding
    num_time_buckets = ts_embeddings.size(1) - 1  # 2049 - 1 = 2048 这里源码应该打错了 size(1)应该是size(0)
    time_bucket_increments = 60.0
    time_bucket_divisor = 1.0
    time_delta = 0
    timestamps = jagged_to_padded_dense(
        values=timestamps.unsqueeze(-1),
        offsets=[seq_offsets],
        max_lengths=[max_seq_len],
        padding_value=0,
    ).squeeze(-1)
    query_time = torch.gather(  # 获取查询时间（序列最后一个时间戳）
        timestamps, dim=1, index=(seq_lengths - 1).unsqueeze(1).clamp(min=0)
    )
    ts = query_time - timestamps    # 计算每个位置距离查询时间的间隔
    ts = ts + time_delta
    ts = ts.clamp(min=1e-6) / time_bucket_increments
    if time_bucket_fn == "log":
        ts = torch.log(ts)
    else:
        ts = torch.sqrt(ts)
    ts = (ts / time_bucket_divisor).clamp(min=0).int()
    ts = torch.clamp(
        ts,
        min=0,
        max=num_time_buckets,
    )
    # 原始时间差（秒）: [3600, 1800, 900, 300, 0]
    # 标准化 (/60):     [60, 30, 15, 5, 0] 
    # 对数变换:         [4.1, 3.4, 2.7, 1.6, 0]
    # 转为整数桶:       [4, 3, 2, 1, 0]
    position_embeddings = torch.index_select(   # 根据位置索引查找对应的位置嵌入向量
        pos_embeddings, 0, pos_inds.reshape(-1)
    ).view(B, max_seq_len, -1)
    time_embeddings = torch.index_select(ts_embeddings, 0, ts.reshape(-1)).view(    # 根据时间桶索引查找对应的时间嵌入向量
        B, max_seq_len, -1
    )
    # 合并嵌入
    return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        seq_embeddings,
        [seq_offsets],
        (time_embeddings + position_embeddings).to(seq_embeddings.dtype),
    )[0]
