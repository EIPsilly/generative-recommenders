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

from typing import Dict, List, Optional, Tuple

import torch

from generative_recommenders.common import HammerModule
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged


class ActionEncoder(HammerModule):
    def __init__(
        self,
        action_embedding_dim: int,  # 8
        action_feature_name: str,    # "action_weight" KuaiRand
        action_weights: List[int],  # [1, 2, 4, 8, 16, 32, 64, 128]
        watchtime_feature_name: str = "",
        watchtime_to_action_thresholds_and_weights: Optional[
            List[Tuple[int, int]]
        ] = None,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._watchtime_feature_name: str = watchtime_feature_name
        self._action_feature_name: str = action_feature_name    # "action_weight"
        self._watchtime_to_action_thresholds_and_weights: List[Tuple[int, int]] = ( #[]
            watchtime_to_action_thresholds_and_weights
            if watchtime_to_action_thresholds_and_weights is not None
            else []
        )
        self.register_buffer(   # tensor([1, 2, 4, 8, 16, 32, 64, 128])
            "_combined_action_weights",
            torch.tensor(
                action_weights
                + [x[1] for x in self._watchtime_to_action_thresholds_and_weights]
            ),
        )
        self._num_action_types: int = len(action_weights) + len(    # 8 + 0 = 8
            self._watchtime_to_action_thresholds_and_weights
        )
        self._action_embedding_dim = action_embedding_dim
        # 动作嵌入表：每种动作类型对应一个8维嵌入向量
        self._action_embedding_table: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((self._num_action_types, action_embedding_dim)).normal_(
                mean=0, std=0.1
            ),
        )
        # 候选物品的动作嵌入（64维）。主要是为了区分"已发生"和"待预测"的行为。同时保证序列的一致性。
        # 返回的action embedding会直接加在已有的output_embedding上。对应preprocessors.py中的这部分代码
        # output_seq_embeddings = output_seq_embeddings + self._action_embedding_mlp(
        #     action_embeddings
        # )
        self._target_action_embedding_table: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((1, self._num_action_types * action_embedding_dim)).normal_(
                mean=0, std=0.1
            ),
        )

    @property
    def output_embedding_dim(self) -> int:
        return self._action_embedding_dim * self._num_action_types

    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        uih_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        seq_embeddings: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # 例如：[7, 3, 65, 130] 表示4个用户交互的动作组合
        seq_actions = seq_payloads[self._action_feature_name]
        if len(self._watchtime_to_action_thresholds_and_weights) > 0:
            watchtimes = seq_payloads[self._watchtime_feature_name]
            for threshold, weight in self._watchtime_to_action_thresholds_and_weights:
                seq_actions = torch.bitwise_or(
                    seq_actions, (watchtimes >= threshold).to(torch.int64) * weight
                )
        # 将动作组合与动作权重进行按位与操作，得到一个布尔张量，表示每个用户进行了哪些交互动作。
        # 使用位掩码一次性处理多种动作类型，节省了数据量
        # 7 & [1, 2, 4, 8, 16, 32, 64, 128] = [1, 2, 4, 0, 0, 0, 0, 0] > 0
        # = [True, True, True, False, False, False, False, False]
        # 最终得到 shape 为 [B, 8] 的张量
        exploded_actions = (
            torch.bitwise_and(
                seq_actions.unsqueeze(-1), self._combined_action_weights.unsqueeze(0)
            )
            > 0
        )
        # 生成加权动作嵌入，返回 shape 为 [B, 64] 的张量。对于第一个动作，会获得前3个嵌入向量，后面5个因为没有动作所谓为0
        action_embeddings = (
            exploded_actions.unsqueeze(-1) * self._action_embedding_table.unsqueeze(0)
        ).view(-1, self._num_action_types * self._action_embedding_dim)
        # 为候选物品添加动作嵌入
        total_targets: int = seq_embeddings.size(0) - action_embeddings.size(0)
        action_embeddings = concat_2D_jagged(
            max_seq_len=max_uih_len + max_targets,
            values_left=action_embeddings,
            values_right=self._target_action_embedding_table.tile(
                total_targets,
                1,
            ),
            max_len_left=max_uih_len,
            max_len_right=max_targets,
            offsets_left=uih_offsets,
            offsets_right=target_offsets,
            kernel=self.hammer_kernel(),
        )
        return action_embeddings
