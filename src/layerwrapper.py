from typing import List

import torch
import torch.nn as nn
from transformers import Conv1D


# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(
        self,
        layer,
        target_token_idxs: List[int] = [],
        attention_masks: List[torch.tensor] = [],
        token_nums: List[int] = [],
        data_types: List[int] = [],
        layer_id=0,
        layer_name="none",
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        if isinstance(self.layer, Conv1D):
            self.rows = layer.weight.data.shape[1]
            self.columns = layer.weight.data.shape[0]
        else:
            self.rows = layer.weight.data.shape[0]
            self.columns = layer.weight.data.shape[1]

        self.target_token_idxs = target_token_idxs
        self.attention_masks = attention_masks
        self.data_types = data_types
        assert len(set(data_types)) <= 2, "Please set data_types under two."
        self.token_nums = token_nums
        if len(set(data_types)) <= 1:
            self.scaler_row = torch.zeros((self.columns), device=self.dev)
            self.nsamples = 0
        else:
            self.scaler_rows = {}
            self.nsamples = {}
            for data_type in set(data_types):
                self.scaler_rows[data_type] = torch.zeros(
                    (self.columns), device=self.dev
                )
                self.nsamples[data_type] = 0

        self.layer_id = layer_id
        self.layer_name = layer_name
        self.index = 0

    def add_batch(self, inp, out):
        if len(self.attention_masks) != 0:
            attention_mask = self.attention_masks[self.index].bool()[0]
            inp = inp[:, attention_mask]

        if len(self.target_token_idxs) != 0:
            target_token_idx = self.target_token_idxs[self.index]
            token_num = self.token_nums[self.index]
            inp = inp[:, target_token_idx + 1 :]
            assert (
                token_num == inp.shape[1]
            ), f"Not match token_num {token_num} and inp {inp.shape[1]}"

        # attention_mask = attention_mask[:, target_token_idx + 1 :]
        # num_mask = attention_mask.sum()
        # inp = inp * attention_mask
        if len(set(self.data_types)) > 1:
            data_type = self.data_types[self.index]
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            tmp = inp.shape[0]
            if isinstance(self.layer, nn.Linear) or isinstance(self.layer, Conv1D):
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = inp.t()

            self.scaler_rows[data_type] *= self.nsamples[data_type] / (
                self.nsamples[data_type] + tmp
            )
            self.nsamples[data_type] += tmp

            inp = inp.type(torch.float32)
            self.scaler_rows[data_type] += (
                torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples[data_type]
            )
        else:
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            tmp = inp.shape[0]
            if isinstance(self.layer, nn.Linear) or isinstance(self.layer, Conv1D):
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = inp.t()

            self.scaler_row *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp

            inp = inp.type(torch.float32)
            self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

        self.index += 1

    def calculate_scaler_row(self):
        if len(set(self.data_types)) > 1:
            base_scaler_row = self.scaler_rows[list(set(self.data_types))[0]]
            sub_scaler_row = self.scaler_rows[list(set(self.data_types))[1]]
            self.scaler_row = base_scaler_row * (base_scaler_row / sub_scaler_row)
