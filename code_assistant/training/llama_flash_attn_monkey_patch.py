# copied from https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py

from typing import List, Optional, Tuple, Union
import logging

import torch
from torch import nn

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from einops import rearrange

from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        raise ValueError("pretraining_tp > 1 is not supported for flash attention")
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]

    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# def forward(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.Tensor] = None,
#     past_key_value: Optional[Tuple[torch.Tensor]] = None,
#     output_attentions: bool = False,
#     use_cache: bool = False,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#     """Input shape: Batch x Time x Channel

#     attention_mask: [bsz, q_len]
#     """
#     bsz, q_len, _ = hidden_states.size()

#     query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#     key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#     value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#     # [bsz, q_len, nh, hd]
#     # [bsz, nh, q_len, hd]

#     kv_seq_len = key_states.shape[-2]
#     assert past_key_value is None, "past_key_value is not supported"

#     cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
#     # [bsz, nh, t, hd]
#     assert not output_attentions, "output_attentions is not supported"
#     assert not use_cache, "use_cache is not supported"

#     # Flash attention codes from
#     # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

#     # transform the data into the format required by flash attention
#     qkv = torch.stack([query_states, key_states, value_states], dim=2)  # [bsz, nh, 3, q_len, hd]
#     qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
#     # We have disabled _prepare_decoder_attention_mask in LlamaModel
#     # the attention_mask should be the same as the key_padding_mask
#     key_padding_mask = attention_mask

#     if key_padding_mask is None:
#         qkv = rearrange(qkv, "b s ... -> (b s) ...")
#         max_s = q_len
#         cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device)
#         output = flash_attn_unpadded_qkvpacked_func(qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True)
#         output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
#     else:
#         nheads = qkv.shape[-2]
#         x = rearrange(qkv, "b s three h d -> b s (three h d)")
#         x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
#         x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
#         output_unpad = flash_attn_unpadded_qkvpacked_func(
#             x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
#         )
#         output = rearrange(
#             pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len),
#             "b s (h d) -> b s h d",
#             h=nheads,
#         )
#     return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, None


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logging.warning(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
