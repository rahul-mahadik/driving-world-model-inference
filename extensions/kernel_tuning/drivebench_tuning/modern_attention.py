"""Native SDPA with Vista's trained projections and original tensor semantics.

This is an explicit port of the attention math in Vista cc9821b4, whose license
is retained in third_party/Vista-LICENSE. It is not the upstream CrossAttention
fallback: that class does not implement Vista's action-control projections.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NativeSDPACrossAttention(nn.Module):
    """State-dict compatible replacement for MemoryEfficientCrossAttention."""

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64,
                 dropout=0.0, zero_init=False, causal=False, add_lora=False,
                 lora_rank=16, lora_scale=1.0, action_control=False,
                 backend=None, **kwargs):
        super().__init__()
        if kwargs or backend is not None:
            raise ValueError("Unsupported native attention constructor options")
        if any(type(x) is not int or x <= 0 for x in (heads, dim_head, query_dim)):
            raise ValueError("Attention dimensions must be positive integers")
        self.heads, self.dim_head = heads, dim_head
        self.causal = bool(causal)
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        if zero_init:
            nn.init.zeros_(self.to_out[0].weight)
            nn.init.zeros_(self.to_out[0].bias)
        self.add_lora = add_lora
        if add_lora:
            self.lora_scale = lora_scale
            for name, dim in (("q", query_dim), ("k", context_dim), ("v", context_dim)):
                down = nn.Linear(dim, lora_rank, bias=False)
                up = nn.Linear(lora_rank, inner_dim, bias=False)
                nn.init.normal_(down.weight, std=1 / lora_rank)
                nn.init.zeros_(up.weight)
                setattr(self, name + "_adapter_down", down)
                setattr(self, name + "_adapter_up", up)
            self.out_adapter_down = nn.Linear(inner_dim, lora_rank, bias=False)
            self.out_adapter_up = nn.Linear(lora_rank, query_dim, bias=False)
            nn.init.normal_(self.out_adapter_down.weight, std=1 / lora_rank)
            nn.init.zeros_(self.out_adapter_up.weight)
        self.action_control = action_control
        if action_control:
            self.context_dim = context_dim
            self.k_adapter_action_control = nn.Linear(128 * 19, inner_dim, bias=False)
            self.v_adapter_action_control = nn.Linear(128 * 19, inner_dim, bias=False)
            nn.init.zeros_(self.k_adapter_action_control.weight)
            nn.init.zeros_(self.v_adapter_action_control.weight)

    def forward(self, x, context=None, mask=None, additional_tokens=None,
                n_times_crossframe_attn_in_self=0, batchify_xformers=False):
        # Upstream rejects arbitrary masks. Do not silently assign a new meaning
        # to a formerly unsupported request. Causal attention has its own flag.
        if mask is not None:
            raise NotImplementedError("Vista arbitrary attention masks are not supported")
        if additional_tokens is not None:
            token_count = additional_tokens.shape[1]
            x = torch.cat((additional_tokens, x), dim=1)
        context = x if context is None else context
        if self.action_control:
            if context.shape[-1] != self.context_dim + 128 * 19:
                raise ValueError("Vista action context must contain exactly 19 x 128 features")
            context, action = context[..., :self.context_dim], context[..., self.context_dim:]
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        if self.add_lora:
            q = q + self.q_adapter_up(self.q_adapter_down(x)) * self.lora_scale
            k = k + self.k_adapter_up(self.k_adapter_down(context)) * self.lora_scale
            v = v + self.v_adapter_up(self.v_adapter_down(context)) * self.lora_scale
        if self.action_control:
            k = k + self.k_adapter_action_control(action)
            v = v + self.v_adapter_action_control(action)
        if n_times_crossframe_attn_in_self:
            n = n_times_crossframe_attn_in_self
            if type(n) is not int or n <= 0 or x.shape[0] % n:
                raise ValueError("Cross-frame group must divide the batch")
            # Same ordered groups as upstream's repeat(k[::n], 'b ... -> (b n) ...').
            k = k[::n].repeat_interleave(n, dim=0)
            v = v[::n].repeat_interleave(n, dim=0)
        batch = q.shape[0]
        q, k, v = [a.reshape(batch, a.shape[1], self.heads, self.dim_head)
                   .transpose(1, 2).contiguous() for a in (q, k, v)]
        # Temporal attention can have a very large effective spatial batch.
        # Keep the original 32768 flattened-head ceiling without calling xFormers.
        chunk = max(1, 32768 // self.heads) if batchify_xformers else batch
        pieces = [F.scaled_dot_product_attention(
            q[i:i + chunk], k[i:i + chunk], v[i:i + chunk],
            dropout_p=0.0, is_causal=self.causal,
        ) for i in range(0, batch, chunk)]
        out = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        out = out.transpose(1, 2).reshape(batch, -1, self.heads * self.dim_head)
        if additional_tokens is not None:
            out = out[:, token_count:]
        result = self.to_out(out)
        if self.add_lora:
            result = result + self.out_adapter_up(self.out_adapter_down(out)) * self.lora_scale
        return result
