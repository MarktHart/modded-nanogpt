import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.pytorch.attention import RotaryPositionEmbedding


class LlamaLayer(te.TransformerLayer):
    def __init__(self, config):
        super().__init__(
            config.n_embd,
            config.n_embd * 4,
            config.n_head,
            bias=config.bias,
            layernorm_epsilon=1e-05,
            hidden_dropout=config.dropout,
            attention_dropout=config.dropout,
            fuse_qkv_params=True,
            self_attn_mask_type="causal",
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            num_gqa_groups=config.n_head // 2,
            init_method=lambda weight: torch.nn.init.normal_(weight, mean=0.0, std=0.02),
            output_layer_init_method=lambda weight: torch.nn.init.normal_(weight, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer)),
        )


class GPTLayer(te.TransformerLayer):
    def __init__(self, config):
        super().__init__(
            config.n_embd,
            config.n_embd * 4,
            config.n_head,
            bias=config.bias,
            layernorm_epsilon=1e-05,
            hidden_dropout=config.dropout,
            attention_dropout=config.dropout,
            fuse_qkv_params=True,
            self_attn_mask_type="causal",
            normalization="LayerNorm",
            activation="gelu",
            attn_input_format="bshd",
            num_gqa_groups=config.n_head,
            init_method=lambda weight: torch.nn.init.normal_(weight, mean=0.0, std=0.02),
            output_layer_init_method=lambda weight: torch.nn.init.normal_(weight, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer)),
        )


@dataclass
class TransformerModelConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    model_class: str = "GPT"


model_str_to_layer_class = {
    "GPT": GPTLayer,
    "LLAMA": LlamaLayer,
}


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([model_str_to_layer_class[config.model_class](config=config) for _ in range(config.n_layer)]),
            ln_f=te.LayerNorm(config.n_embd),
        )

        if config.model_class != "GPT":
            del modules["wpe"]

        self.transformer = nn.ModuleDict(modules)

        if "LLAMA" in self.config.model_class:
            self.register_buffer("rope", RotaryPositionEmbedding(self.config.n_embd//self.config.n_head)(max_seq_len=self.config.block_size), persistent=False)
            # self.rope = RotaryPositionEmbeddingNonFused(self.config.n_embd//self.config.n_head)(max_seq_len=self.config.block_size)
        else:
            self.rope = None

        self.lm_head = te.Linear(config.n_embd, config.vocab_size, bias=False, init_method=lambda weight: torch.nn.init.normal_(weight, mean=0.0, std=0.02))

        if config.model_class == "GPT":
            torch.nn.init.normal_(self.transformer.wpe.weight, mean=0.0, std=0.02)

        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and "LLAMA" not in self.config.model_class:
            n_params -= self.transformer.wpe.weight.numel()
            # n_params -= self.transformer.wte.weight.numel()
        return n_params

    def forward(self, idx, targets, is_first_microbatch=None):
        device = idx.device
        b, t = idx.size()

        if self.config.model_class == "GPT":
            embeddings = self.transformer.wte(idx) + self.transformer.wpe(torch.arange(0, t, dtype=torch.long, device=device))
        else:
            embeddings = self.transformer.wte(idx)

        x = self.transformer.drop(embeddings)
        for block in self.transformer.h:
            x = block(x, rotary_pos_emb=self.rope, is_first_microbatch=is_first_microbatch)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            # logits = x @ self.transformer.wte.weight.T
            logits = self.lm_head(x, is_first_microbatch=is_first_microbatch)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # logits = x[:, -1:, :] @ self.transformer.wte.weight.T
            logits = self.lm_head(x[:, -1:, :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def init_method(weight):
    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
