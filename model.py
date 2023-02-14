""" PyTorch GPT-J model.

Based on:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.models.gptj.modeling_gptj import GPTJBlock


DTYPE = torch.bfloat16


def init_weights(config, module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear,)):
        # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class EmbeddingModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        init_weights(config, self.wte)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Dict[str, Union[torch.Tensor, Tuple]]:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        device = input_ids.device

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=DTYPE)
            attention_mask = (1.0 - attention_mask) * torch.finfo(DTYPE).min

        inputs_embeds = self.drop(self.wte(input_ids))

        return {
            "input_shape": input_shape,
            "hidden_states": inputs_embeds,
            "attention_mask": attention_mask,
        }


@dataclass
class GPTJBlockShardConfig:
    start_block: int
    end_block: int
    includ_layer_norm: bool = False


class GPTJBlocksModule(nn.Module):
    def __init__(self, config, shard_config):
        super().__init__()

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size

        self._start_block = shard_config.start_block
        self._end_block = shard_config.end_block

        self.h = nn.ModuleList(
            [GPTJBlock(config) for _ in range(self._start_block, self._end_block + 1)]
        )

        if shard_config.includ_layer_norm:
            self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        else:
            self.ln_f = None

    def forward(
        self,
        input_shape: Tuple[int],
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Dict[str, Union[torch.Tensor, None]]:
        output_shape = input_shape + (hidden_states.size(-1),)

        for i, block in enumerate(self.h):
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = outputs[0]

        if self.ln_f:
            hidden_states = self.ln_f(hidden_states).view(output_shape)
            # Output only hidden_states for last LMHead model.
            return {
                "hidden_states": hidden_states,
            }
        else:
            # Have more attention blocks to run.
            # Output everything including input_shape and attention_mask.
            return {
                "input_shape": input_shape,
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
            }


class LMHeadModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        init_weights(config, self.lm_head)

    def forward(
        self,
        hidden_states,
        labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, Union[torch.Tensor, None]]:
        # Compute logits.
        logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)

        return {"loss": loss, "logits": logits}
