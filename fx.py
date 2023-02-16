"""Test to see if torch fx tracing works for GPT-J models.
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from transformers import AutoConfig

from model import EmbeddingModule, GPTJBlocksModule, GPTJBlockShardConfig, LMHeadModule


MODEL_DIR = ""  # Pre-downloaded GPT-J model directory.


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(MODEL_DIR)

        self._embedding = EmbeddingModule(config)
        self._h = GPTJBlocksModule(
            config, GPTJBlockShardConfig(0, 5, includ_layer_norm=True)
        )
        self._lm_head = LMHeadModule(config)

    def forward(self, input_ids, attention_mask):
        inputs = self._embedding(input_ids, attention_mask)
        inputs = self._h(**inputs)
        return self._lm_head(**inputs)


def trace():
    model = Model()

    # Tracing fails at positional encoding step?
    #     TypeError: arange() received an invalid combination of arguments -
    #         got (int, Proxy, device=Attribute, dtype=torch.dtype) ...
    #     position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
    symbolic_traced = symbolic_trace(model)

    # !!!
    print(symbolic_traced.graph)
    
    # code?!
    print(symbolic_traced.code)


if __name__ == "__main__":
    trace()
