# copied from https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py

from typing import List, Optional, Tuple, Union

import torch
import transformers
from einops import rearrange
from flash_attn import flash_attn_func


def forward(
    self,
    hidden_states: torch.Tensor,
    layer_past: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor, Optional[torch.Tensor]],
    Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
]:
    if encoder_hidden_states is not None:
        if not hasattr(self, "q_attn") or not self.is_cross_attention:
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`."
            )
        query = self.q_attn(hidden_states)
        key_value = self.c_attn(encoder_hidden_states)
        attention_mask = encoder_attention_mask
    elif self.multi_query:
        query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=2)
    else:
        # Note: We split as (self.num_heads, 3, self.head_dim) instead of (3, self.num_heads, self.head_dim),
        # i.e., the memory layout is not the same as GPT2.
        # This makes the concatenation with past_key_value more efficient.
        query, key_value = (
            self.c_attn(hidden_states)
            .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
            .transpose(1, 2)
            .split((self.head_dim, 2 * self.head_dim), dim=3)
        )
    if layer_past is not None:
        key_value = torch.cat((layer_past, key_value), dim=-2)
    present = key_value if use_cache else None

    key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

    if self.multi_query:
        batch_size, query_length, _ = query.shape
        query = query.reshape(batch_size, query_length, self.num_heads, self.head_dim)
        key, value = [torch.unsqueeze(x, 2) for x in [key, value]]
    else:
        query, key, value = [rearrange(x, "b h s d -> b s h d") for x in [query, key, value]]
    query, key, value = [x.to(torch.bfloat16) for x in [query, key, value]]
    # print(f"{query.shape=} {key.shape=} {value.shape=}")
    attn_output = flash_attn_func(query, key, value, dropout_p=self.attn_dropout.p, causal=True)
    attn_output = self.c_proj(attn_output.reshape(hidden_states.shape))
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        raise NotImplementedError("`output_attentions` is not supported when `use_flash_attn` is True")
    return outputs  # a, present, (attentions)


def replace_starcoder_attn_with_flash_attn():
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeAttention.forward = forward
