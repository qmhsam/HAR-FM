# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F

# Modified based on https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/wav2vec2

class Quantizer(nn.Module):
    def __init__(self, input_dim, num_groups, num_vars, vq_dim=96, tau_range=(2.0, 0.5, 0.999995)):
        super(Quantizer, self).__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.num_vars = num_vars

        self.var_dim = vq_dim // self.num_groups

        # weight_proj: can be some @FwBlock with activation function
        #               simple indices prediction.
        self.weight_proj = nn.Linear(self.input_dim, self.num_groups * self.num_vars) 
        nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
        nn.init.zeros_(self.weight_proj.bias)

        # Initialize the codebook
        # do not combine groups
        self.vars = nn.Parameter(torch.randn(1, num_groups * num_vars, self.var_dim))

        self.max_temp, self.min_temp, self.temp_decay = tau_range
        self.curr_temp = self.max_temp

        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.num_groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if self.num_groups != 1:
                self.codebook_indices = self.codebook_indices.view(
                    self.num_vars ** self.num_groups, -1
                )
                for b in range(1, self.num_groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices
    
    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.num_groups):
            exponent = self.num_groups - i - 1
            res += indices[..., i] * (self.num_vars ** exponent)
        return res
    
    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            self.vars.squeeze(0)
            .index_select(0, indices)
            .view(self.num_vars ** self.num_groups, -1)
        )
    
    def forward(self, x):
        # x shape: [batch_size, time_length, input_dim]
        bsz, tsz, fsz = x.shape 
        x   = self.weight_proj(x.reshape(-1, fsz))
        x   = x.view(bsz * tsz * self.num_groups, -1) # reshape to indices

        # grab the indices from each group
        _, k = x.max(-1)

        # obtain the one_hot encoding
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz * self.num_groups, -1)
        ) 
        
        # hard_probs = torch.mean(hard_x.float(), dim=0)
        # code_perplexity = torch.exp(
        #     -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        # ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.num_groups, -1).float(), dim=-1
        ).mean(dim=0)
        prob_perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()
        
        # Each sample would select @num_groups entries from each codebook group.
        # indices: 
        #   permutation of [group_id, entries]

        # Gumbel-Softmax for differentiable argmin
        # x: [bsz * tsz * self.num_groups, self.num_vars]
        # if self.training:
        x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x) # grab the trainable indices
        # else:
        #     x = hard_x # if 

        # x: [bsz * tsz, self.num_groups * self.num_vars]
        x = x.view(bsz * tsz, -1) # retrieve the shape

        # # @indices
        # grab the indices < discrete indices
        indices = (
            x.view(bsz * tsz * self.num_groups, -1)
            .argmax(dim=-1)
            .view(bsz, tsz, self.num_groups)
            .detach()
        ) 

        # # Get quantized vectors
        # [bsz * tsz, num_groups * num_vars, 1] x [1, num_groups * num_vars, self.var_dim]
        # -> x: [bsz * tsz, num_groups * num_vars, var_dim]
        x = x.unsqueeze(-1) * self.vars
        # -> x: [bsz * tsz, num_groups, num_vars, var_dim]
        x = x.view(bsz * tsz, self.num_groups, self.num_vars, -1) 
        # -> x: [bsz * tsz, num_groups, var_dim]
        x = x.sum(-2) # when training, all elements will be reached
        x = x.view(bsz, tsz, -1)

        result_info = {
            "curr_temp": self.curr_temp,
            "num_vars": self.num_vars * self.num_groups,
            "prob_perplexity": prob_perplexity
        }

        # > perplexity loss panelize the elements usage.
        # > l_p = (result_info["num_vars"] - result_info["prob_perplexity"]) / result_info["num_vars"]

        return x, indices, result_info
