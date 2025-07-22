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

# pyre-unsafe

import torch
from typing import List, Tuple


def batch_gather_embeddings(
    rowwise_indices: torch.Tensor,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        rowwise_indices: (B, N) x int, where each entry is in [0, X).
        embeddings: (B, X, D,) x float.

    Returns:
        (B, N, D,) x float, embeddings corresponding to rowwise_indices.
    """
    _, N = rowwise_indices.size()
    B, X, D = embeddings.size()
    flattened_indices = (
        rowwise_indices
        + torch.arange(
            start=0,
            end=B,
            step=1,
            dtype=rowwise_indices.dtype,
            device=rowwise_indices.device,
        )
        .unsqueeze(1)
        .expand(-1, N)
        * X
    )
    return embeddings.view(-1, D)[flattened_indices, :].reshape(
        rowwise_indices.size() + (D,)
    )


def batch_scatter_embeddings(
    dst_embeddings: torch.Tensor,
    rowwise_indices: torch.Tensor,
    src_embeddings: torch.Tensor,
) -> None:
    """
    Args:
        dst_embeddings: (B, N, D,) x float.
        rowwise_indices: (B,) x int, where each entry is in [0, N - 1).
        source_embeddings: (B, D,) x float.
    """
    B, N, D = dst_embeddings.size()
    flattened_indices = rowwise_indices + torch.arange(
        start=0,
        end=B * N,
        step=N,
        dtype=rowwise_indices.dtype,
        device=rowwise_indices.device,
    )
    dst_embeddings.view(B * N, D)[flattened_indices, :] = src_embeddings


def get_current_embeddings(
    lengths: torch.Tensor,
    encoded_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        lengths: (B,) x int
        seq_embeddings: (B, N, D,) x float

    Returns:
        (B, D,) x float, where [i, :] == encoded_embeddings[i, lengths[i] - 1, :]
    """
    B, N, D = encoded_embeddings.size()
    # 假设：B=3, N=5, lengths=[3, 2, 4]
    # 
    # lengths - 1 = [2, 1, 3]  # 每个序列的最后有效位置（0-indexed）
    # torch.arange(0, 3) * N = [0, 5, 10]  # 每个样本在展平数组中的起始偏移
    # 
    # flattened_offsets = [2, 1, 3] + [0, 5, 10] = [2, 6, 13]
    flattened_offsets = (lengths - 1) + torch.arange(
        start=0, end=B, step=1, dtype=lengths.dtype, device=lengths.device
    ) * N
    # 原始形状: (3, 5, D)
    # [
    #   [[emb_0_0], [emb_0_1], [emb_0_2], [pad], [pad]],     # 用户0，长度=3
    #   [[emb_1_0], [emb_1_1], [pad], [pad], [pad]],         # 用户1，长度=2  
    #   [[emb_2_0], [emb_2_1], [emb_2_2], [emb_2_3], [pad]] # 用户2，长度=4
    # ]
    # 展平后: (15, D)
    # 索引: [2, 6, 13] 对应 [emb_0_2, emb_1_1, emb_2_3]
    #
    # 结果: (3, D) 每个用户的最后有效嵌入
    return encoded_embeddings.reshape(-1, D)[flattened_offsets, :].reshape(B, D)


# 将每个序列的长度（lengths）转换为偏移量（offsets），即计算累积和并在最前面添加 0。  [4, 5, 6] -> [0, 4, 9, 15]
def _asynchronous_complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    """
    CPU-compatible replacement for torch.ops.fbgemm.asynchronous_complete_cumsum.
    Computes cumulative sum of lengths and prepends 0.
    
    Args:
        lengths: Tensor of shape [B] containing sequence lengths
        
    Returns:
        Tensor of shape [B+1] containing cumulative offsets
    """
    return torch.cat([torch.zeros(1, device=lengths.device, dtype=lengths.dtype), 
                     torch.cumsum(lengths, dim=0)])

# 将 稀疏表示（jagged） 转换为 稠密张量（dense） [12905, 50] -> [128, 211, 50]
def _jagged_to_padded_dense(values: torch.Tensor, offsets: List[torch.Tensor], 
                           max_lengths: List[int], padding_value: float = 0.0) -> torch.Tensor:
    """
    CPU-compatible replacement for torch.ops.fbgemm.jagged_to_padded_dense.
    Converts jagged tensor to padded dense tensor.
    
    Args:
        values: Tensor of shape [total_elements, feature_dim]
        offsets: List containing one tensor of shape [batch_size + 1]
        max_lengths: List containing maximum sequence length
        padding_value: Value to use for padding
        
    Returns:
        Padded dense tensor of shape [batch_size, max_length, feature_dim]
    """
    x_offsets = offsets[0]
    max_length = max_lengths[0]
    batch_size = x_offsets.size(0) - 1
    feature_dim = values.size(1)
    
    # Create padded tensor
    padded = torch.full((batch_size, max_length, feature_dim), 
                       padding_value, 
                       device=values.device, 
                       dtype=values.dtype)
    
    # Fill in the actual values
    for i in range(batch_size):
        start_idx = x_offsets[i]
        end_idx = x_offsets[i + 1]
        seq_length = end_idx - start_idx
        if seq_length > 0:
            padded[i, :seq_length] = values[start_idx:end_idx]
    
    return padded

# 将 稠密张量 转换回 稀疏表示（jagged tensor）
# 根据 offsets 指示的长度，从 dense tensor 中提取每段有效数据，拼接成 jagged 格式（返回一个元组） [128, 211, 50] -> [12905, 50]
def _dense_to_jagged(dense_tensor: torch.Tensor, offsets: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    """
    CPU-compatible replacement for torch.ops.fbgemm.dense_to_jagged.
    Converts padded dense tensor to jagged tensor.
    
    Args:
        dense_tensor: Tensor of shape [batch_size, max_length, feature_dim]
        offsets: List containing one tensor of shape [batch_size + 1]
        
    Returns:
        Tuple containing jagged tensor of shape [total_elements, feature_dim]
    """
    x_offsets = offsets[0]
    batch_size = x_offsets.size(0) - 1
    feature_dim = dense_tensor.size(-1)
    
    # Calculate total elements
    total_elements = int(x_offsets[-1].item())
    
    # Create jagged tensor
    jagged = torch.zeros(total_elements, feature_dim, 
                        device=dense_tensor.device, 
                        dtype=dense_tensor.dtype)
    
    # Fill in the actual values
    for i in range(batch_size):
        start_idx = x_offsets[i]
        end_idx = x_offsets[i + 1]
        seq_length = end_idx - start_idx
        if seq_length > 0:
            jagged[start_idx:end_idx] = dense_tensor[i, :seq_length]
    
    return (jagged,)