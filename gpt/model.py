"""Token-agnostic GPT (nanoGPT-style) with optional Multi-Token Prediction (MTP).

MTP heads predict several future tokens from the same hidden state (training
signal + draft tokens). Speculative decoding verifies drafts with the main head
so accepted tokens match standard autoregressive sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False
    # Extra future tokens predicted by MTP heads (0 = plain next-token LM).
    n_mtp: int = 3
    mtp_loss_weight: float = 0.3


def config_from_dict(d: dict) -> GPTConfig:
    """Build GPTConfig from a checkpoint dict (ignores unknown keys)."""
    allowed = {f.name for f in fields(GPTConfig)}
    kwargs = {k: v for k, v in d.items() if k in allowed}
    # Old checkpoints had no MTP fields.
    kwargs.setdefault("n_mtp", 0)
    kwargs.setdefault("mtp_loss_weight", 0.3)
    return GPTConfig(**kwargs)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MTPHead(nn.Module):
    """Predicts one future token (offset >= 2) from a hidden state."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.gelu(self.proj(self.ln(x))))


def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = 40,
) -> torch.Tensor:
    """Sample token ids from [B, V] logits → [B, 1]."""
    logits = logits / max(temperature, 1e-6)
    if top_k is not None:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.clone()
        logits[logits < values[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _expand_linear_vocab(linear: nn.Linear, new_vocab_size: int, n_embd: int) -> nn.Linear:
    old = linear.out_features
    device = linear.weight.device
    dtype = linear.weight.dtype
    old_w = linear.weight.detach()
    new_w = torch.empty(new_vocab_size, n_embd, device=device, dtype=dtype)
    nn.init.normal_(new_w, mean=0.0, std=0.02)
    new_w[:old].copy_(old_w)
    out = nn.Linear(n_embd, new_vocab_size, bias=False).to(device)
    out.weight = nn.Parameter(new_w)
    return out


class CharGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying for main next-token head
        self.transformer.wte.weight = self.lm_head.weight
        self.mtp_heads = nn.ModuleList(
            [MTPHead(config) for _ in range(config.n_mtp)]
        )
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(
                f"Sequence length {t} exceeds block_size {self.config.block_size}"
            )
        pos = torch.arange(0, t, device=idx.device, dtype=torch.long)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        return self.transformer.ln_f(x)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.forward_hidden(idx)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # MTP: head k at position t predicts targets[t + k + 1]? 
            # targets[t] is already next token (offset +1). Head k predicts offset +(k+2)
            # i.e. targets[t + (k+1)].
            if self.config.n_mtp > 0 and self.config.mtp_loss_weight > 0:
                mtp_losses = []
                for k, head in enumerate(self.mtp_heads):
                    shift = k + 1  # extra steps beyond the main next-token
                    if shift >= targets.size(1):
                        break
                    mtp_logits = head(h[:, : targets.size(1) - shift, :])
                    mtp_tgt = targets[:, shift:]
                    mtp_losses.append(
                        F.cross_entropy(
                            mtp_logits.reshape(-1, mtp_logits.size(-1)),
                            mtp_tgt.reshape(-1),
                        )
                    )
                if mtp_losses:
                    loss = loss + self.config.mtp_loss_weight * sum(mtp_losses) / len(
                        mtp_losses
                    )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 40,
        speculative: bool | None = None,
    ) -> torch.Tensor:
        use_spec = (
            speculative
            if speculative is not None
            else (self.config.n_mtp > 0)
        )
        if use_spec and self.config.n_mtp > 0:
            return self.generate_speculative(
                idx, max_new_tokens, temperature=temperature, top_k=top_k
            )
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            next_id = _sample_from_logits(logits[:, -1, :], temperature, top_k)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    @torch.no_grad()
    def generate_speculative(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 40,
    ) -> torch.Tensor:
        """MTP draft + main-head verification (self-speculative decoding).

        Drafts up to 1 + n_mtp tokens per outer step; verifies with the main
        lm_head so accepted tokens match the standard AR distribution.
        """
        n_mtp = self.config.n_mtp
        generated = 0
        while generated < max_new_tokens:
            idx_cond = idx[:, -self.config.block_size :]
            h = self.forward_hidden(idx_cond)
            main_logits = self.lm_head(h[:, -1, :])

            draft_tokens = [_sample_from_logits(main_logits, temperature, top_k)]
            # Store draft distributions used for acceptance ratios.
            def _probs(logits: torch.Tensor) -> torch.Tensor:
                scaled = logits / max(temperature, 1e-6)
                if top_k is not None:
                    values, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled = scaled.clone()
                    scaled[scaled < values[:, [-1]]] = float("-inf")
                return F.softmax(scaled, dim=-1)

            draft_probs = [_probs(main_logits)]
            h_last = h[:, -1, :]
            max_draft = min(1 + n_mtp, max_new_tokens - generated)
            for k in range(max_draft - 1):
                mtp_logits = self.mtp_heads[k](h_last)
                draft_tokens.append(_sample_from_logits(mtp_logits, temperature, top_k))
                draft_probs.append(_probs(mtp_logits))

            draft = torch.cat(draft_tokens, dim=1)  # [B, gamma]
            gamma = draft.size(1)

            if gamma == 1:
                idx = torch.cat([idx, draft], dim=1)
                generated += 1
                continue

            # One verify forward: prefix = context + draft[:-1]
            prefix = torch.cat([idx_cond, draft[:, :-1]], dim=1)
            prefix = prefix[:, -self.config.block_size :]
            p_h = self.forward_hidden(prefix)
            target_logits = self.lm_head(p_h[:, -gamma:, :])  # [B, gamma, V]

            # Accept draft tokens left-to-right; reject → residual resample.
            n_accept = 0
            resampled: torch.Tensor | None = None
            for i in range(gamma):
                tok = draft[:, i : i + 1]  # [B, 1]
                q = draft_probs[i]
                p = _probs(target_logits[:, i, :])
                q_tok = q.gather(1, tok).squeeze(1)
                p_tok = p.gather(1, tok).squeeze(1)
                ratio = (p_tok / q_tok.clamp_min(1e-8)).clamp(max=1.0)
                if bool((torch.rand_like(ratio) <= ratio).all()):
                    n_accept += 1
                    continue
                residual = (p - q).clamp_min(0.0)
                mass = residual.sum(dim=-1, keepdim=True)
                if bool((mass > 0).all()):
                    residual = residual / mass.clamp_min(1e-8)
                    resampled = torch.multinomial(residual, num_samples=1)
                else:
                    resampled = _sample_from_logits(
                        target_logits[:, i, :], temperature, top_k
                    )
                break

            if resampled is not None:
                idx = torch.cat([idx, draft[:, :n_accept], resampled], dim=1)
                generated += n_accept + 1
            else:
                idx = torch.cat([idx, draft], dim=1)
                generated += gamma

        return idx

    def expand_vocab(self, new_vocab_size: int) -> None:
        """Grow token embedding / lm_head / MTP outs; keep existing rows."""
        old = self.config.vocab_size
        if new_vocab_size == old:
            return
        if new_vocab_size < old:
            raise ValueError(
                f"Cannot shrink vocab from {old} to {new_vocab_size}"
            )
        self.lm_head = _expand_linear_vocab(
            self.lm_head, new_vocab_size, self.config.n_embd
        )
        self.transformer.wte = nn.Embedding(
            new_vocab_size, self.config.n_embd
        ).to(self.lm_head.weight.device)
        self.transformer.wte.weight = self.lm_head.weight
        for head in self.mtp_heads:
            head.out = _expand_linear_vocab(
                head.out, new_vocab_size, self.config.n_embd
            )
        self.config.vocab_size = new_vocab_size

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
