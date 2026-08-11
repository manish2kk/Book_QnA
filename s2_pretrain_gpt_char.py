#!/usr/bin/env python3
"""Pretrain a 6-layer character-level GPT on a book text file.

Use --resume with the same --out-dir to continue from a previous checkpoint
(e.g. train book 2 without wiping book 1 weights).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from gpt.model import CharGPT, GPTConfig, config_from_dict
from gpt.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "book1_txt" / "KUNDALINI YOGA.txt"
DEFAULT_OUT = ROOT / "gpt_checkpoints" / "kundalini_char_gpt"
DEFAULT_VOCAB_DIR = ROOT / "book1_txt"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def list_vocab_files(vocab_dir: Path) -> list[Path]:
    files = sorted(vocab_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {vocab_dir}")
    return files


def find_checkpoint(out_dir: Path) -> Path:
    # Prefer last.pt so --resume continues from the latest step (best.pt may
    # be older if val loss got worse later).
    for name in ("last.pt", "best.pt"):
        path = out_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No best.pt/last.pt in {out_dir}. Train once without --resume first."
    )


def make_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: CharGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    eval_iters: int = 50,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_iters):
            xb, yb = make_batch(data, batch_size, block_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        default=DEFAULT_VOCAB_DIR,
        help="Build character vocab from all *.txt in this folder (default: book1_txt).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load best/last checkpoint from --out-dir and continue training "
        "(keeps previous weights; expands vocab if needed).",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--n-mtp",
        type=int,
        default=3,
        help="Multi-token prediction heads (0 disables MTP / speculative drafts).",
    )
    parser.add_argument(
        "--mtp-loss-weight",
        type=float,
        default=0.3,
        help="Weight for MTP auxiliary losses relative to next-token loss.",
    )
    parser.add_argument("--step-sleep", type=float, default=1.0,
        help="Seconds to pause after each training step (cools laptop; 0 = no pause).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    text = args.data.read_text(encoding="utf-8")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.out_dir / "vocab.txt"
    vocab_files = list_vocab_files(args.vocab_dir)
    print(
        f"Vocab sources ({len(vocab_files)} files in {args.vocab_dir.name}): "
        + ", ".join(p.name for p in vocab_files)
    )

    start_step = 0
    data_history: list[str] = []
    optimizer_state = None
    vocab_grew = False

    if args.resume:
        ckpt_path = find_checkpoint(args.out_dir)
        if not vocab_path.exists():
            raise FileNotFoundError(f"Missing vocab at {vocab_path}")
        tokenizer = CharTokenizer.load(vocab_path)
        n_new = tokenizer.extend_with_files(vocab_files)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = config_from_dict(ckpt["config"])
        has_mtp = any(k.startswith("mtp_heads") for k in ckpt["model"])
        if not has_mtp:
            config.n_mtp = 0
        # Architecture comes from the checkpoint so weights still match.
        model = CharGPT(config).to(device)
        model.load_state_dict(ckpt["model"], strict=False)
        if n_new:
            print(f"Vocab grew by {n_new} chars → {tokenizer.vocab_size}")
        # Also covers a prior crash that saved vocab.txt before weights.
        if tokenizer.vocab_size > model.config.vocab_size:
            old_vs = model.config.vocab_size
            vocab_grew = True
            model.expand_vocab(tokenizer.vocab_size)
            print(f"Expanded embeddings {old_vs} → {tokenizer.vocab_size}")
        elif tokenizer.vocab_size < model.config.vocab_size:
            raise ValueError(
                f"vocab.txt size {tokenizer.vocab_size} < checkpoint "
                f"vocab {model.config.vocab_size}; refuse to shrink"
            )
        start_step = int(ckpt.get("step", 0))
        # Adam moments are sized to old embeddings; skip after vocab growth.
        if not vocab_grew:
            optimizer_state = ckpt.get("optimizer")
        prev = ckpt.get("data_files") or (
            [ckpt["data_file"]] if ckpt.get("data_file") else []
        )
        data_history = list(prev)
        print(
            f"Resumed {ckpt_path.name} (step={start_step}, "
            f"val={ckpt.get('val_loss')}, params kept)"
        )
    else:
        tokenizer = CharTokenizer.from_files(vocab_files)
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            n_mtp=args.n_mtp,
            mtp_loss_weight=args.mtp_loss_weight,
        )
        model = CharGPT(config).to(device)
        print(
            f"Starting from scratch (new random weights), "
            f"vocab={tokenizer.vocab_size}, n_mtp={config.n_mtp}"
        )

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = len(data)
    split = int(n * (1.0 - args.val_ratio))
    train_data, val_data = data[:split], data[split:]
    print(
        f"Corpus: {args.data.name} | chars={n:,} | vocab={tokenizer.vocab_size} | "
        f"train={len(train_data):,} | val={len(val_data):,}"
    )
    print(f"Model params: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    if vocab_grew:
        print("Using fresh optimizer (vocab grew; Adam state shapes changed)")
    elif optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("Restored optimizer state")
        except (ValueError, RuntimeError) as exc:
            print(f"Optimizer state not restored ({exc}); using fresh optimizer")

    data_history.append(str(args.data))
    tokenizer.save(vocab_path)
    best_val = math.inf
    t0 = time.time()
    t_last = t0

    for local_step in range(1, args.max_iters + 1):
        step = start_step + local_step
        xb, yb = make_batch(train_data, args.batch_size, config.block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if args.step_sleep > 0:
            time.sleep(args.step_sleep)

        if local_step % 50 == 0 or local_step == 1:
            now = time.time()
            print(
                f"step {step:5d} (+{local_step}/{args.max_iters})  "
                f"loss={loss.item():.4f}  (+{now - t_last:.1f}s, total {now - t0:.1f}s)"
            )
            t_last = now

        if local_step % args.eval_interval == 0 or local_step == args.max_iters:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                config.block_size,
                device,
            )
            now = time.time()
            print(
                f"eval @ {step}: train={losses['train']:.4f} val={losses['val']:.4f} "
                f"(+{now - t_last:.1f}s, total {now - t0:.1f}s)"
            )
            t_last = now
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.__dict__,
                "step": step,
                "val_loss": losses["val"],
                "data_file": str(args.data),
                "data_files": data_history,
            }
            torch.save(ckpt, args.out_dir / "last.pt")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(ckpt, args.out_dir / "best.pt")
                print(f"  saved best.pt (val={best_val:.4f})")

    meta = {
        "data_file": str(args.data),
        "data_files": data_history,
        "vocab_dir": str(args.vocab_dir),
        "vocab_files": [p.name for p in vocab_files],
        "vocab_size": tokenizer.vocab_size,
        "block_size": config.block_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "n_mtp": config.n_mtp,
        "mtp_loss_weight": config.mtp_loss_weight,
        "max_iters": args.max_iters,
        "total_steps": start_step + args.max_iters,
        "resumed": args.resume,
        "best_val_loss": best_val,
        "device": str(device),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done. Checkpoints in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
