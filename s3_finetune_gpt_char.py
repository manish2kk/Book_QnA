#!/usr/bin/env python3
"""Fine-tune a pretrained word-level GPT on Question–Answer instruction pairs.

Loads weights from a pretrained checkpoint (--init-from), trains on
qa_finetune/qa_pairs.jsonl (or --data), and writes a new checkpoint to --out-dir.

Example:
  .venv/bin/python finetune_gpt_word.py --step-sleep 0 --max-iters 500
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from gpt.model import CharGPT, config_from_dict
from gpt.word_tokenizer import WordTokenizer

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "qa_finetune" / "qa_pairs.jsonl"
DEFAULT_INIT = ROOT / "gpt_checkpoints" / "yoga_word_gpt"
DEFAULT_OUT = ROOT / "gpt_checkpoints" / "yoga_word_qa"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_checkpoint(ckpt_dir: Path) -> Path:
    for name in ("best.pt", "last.pt"):
        path = ckpt_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No best.pt/last.pt in {ckpt_dir}. Pretrain first with pretrain_gpt_word.py"
    )


def load_qa_pairs(path: Path) -> list[dict]:
    pairs: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "question" not in obj or "answer" not in obj:
            raise ValueError(f"{path}:{line_no}: need 'question' and 'answer' fields")
        pairs.append(obj)
    if not pairs:
        raise ValueError(f"No QA pairs found in {path}")
    return pairs


def format_example(obj: dict) -> str:
    instruction = obj.get(
        "instruction",
        "Answer the question using yoga knowledge from the books.",
    )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Question:\n{obj['question'].strip()}\n\n"
        f"### Answer:\n{obj['answer'].strip()}\n\n"
    )


def build_corpus(pairs: list[dict]) -> str:
    return "".join(format_example(p) for p in pairs)


def encode_with_answer_mask(
    tokenizer: WordTokenizer, text: str
) -> tuple[list[int], int]:
    """
    Encode text and return (token_ids, ignore_until).
    Targets for positions before ignore_until are masked (-100) so the model
    mainly learns to generate the answer.
    """
    marker = "### Answer:\n"
    pos = text.find(marker)
    all_ids = tokenizer.encode(text)
    if pos < 0 or len(all_ids) < 2:
        return all_ids, 0

    prefix = text[: pos + len(marker)]
    prefix_ids = tokenizer.encode(prefix)
    # Ignore predicting tokens still inside the instruction/question prefix.
    ignore_until = max(0, len(prefix_ids) - 1)
    return all_ids, ignore_until


def make_masked_batch(
    examples: list[tuple[list[int], int]],
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample QA examples, truncate/pad to block_size, apply answer-only loss mask."""
    # Use unk/space-like last resort: pad with first token id 0 (<unk> usually).
    pad_id = 0
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for _ in range(batch_size):
        ids, ignore_until = examples[torch.randint(0, len(examples), (1,)).item()]
        # Need block_size+1 tokens for x/y; truncate from the start if too long
        # so the answer (near the end) is kept.
        if len(ids) > block_size + 1:
            ids = ids[-(block_size + 1) :]
            # After left-truncation, ignore_until no longer applies cleanly:
            # train on all remaining tokens.
            ignore_until = 0
        if len(ids) < 2:
            ids = ids + [pad_id, pad_id]
        x = ids[:-1]
        y = ids[1:]
        # Mask targets that only predict instruction/question tokens.
        y_masked = []
        for i, tok in enumerate(y):
            # i is position in x; predicting y[i] == ids[i+1]
            if i < ignore_until:
                y_masked.append(-100)
            else:
                y_masked.append(tok)
        # Pad to block_size
        if len(x) < block_size:
            pad_n = block_size - len(x)
            x = x + [pad_id] * pad_n
            y_masked = y_masked + [-100] * pad_n
        else:
            x = x[:block_size]
            y_masked = y_masked[:block_size]
        xs.append(torch.tensor(x, dtype=torch.long))
        ys.append(torch.tensor(y_masked, dtype=torch.long))
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


@torch.no_grad()
def estimate_loss(
    model: CharGPT,
    examples: list[tuple[list[int], int]],
    batch_size: int,
    block_size: int,
    device: torch.device,
    eval_iters: int = 40,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = make_masked_batch(examples, batch_size, block_size, device)
        logits, _ = model(xb)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.view(-1),
            ignore_index=-100,
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="QA JSONL file")
    parser.add_argument(
        "--init-from",
        type=Path,
        default=DEFAULT_INIT,
        help="Pretrained word GPT checkpoint dir to fine-tune from",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Where to save the fine-tuned checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Lower than pretrain by default for fine-tuning",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--step-sleep",
        type=float,
        default=1.0,
        help="Seconds to pause after each step (0 = full speed)",
    )
    parser.add_argument(
        "--full-loss",
        action="store_true",
        help="Train on full sequence (not answer-only). Default: answer tokens only.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    pairs = load_qa_pairs(args.data)
    print(f"Loaded {len(pairs)} QA pairs from {args.data}")

    ckpt_path = find_checkpoint(args.init_from)
    vocab_path = args.init_from / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab at {vocab_path}")

    tokenizer = WordTokenizer.load(vocab_path)
    corpus = build_corpus(pairs)
    n_new = tokenizer.extend_with_tokens(WordTokenizer.tokenize(corpus))
    if n_new:
        print(f"Vocab grew by {n_new} tokens → {tokenizer.vocab_size}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = config_from_dict(ckpt["config"])
    has_mtp = any(k.startswith("mtp_heads") and ".out.weight" in k for k in ckpt["model"])
    if not has_mtp:
        config.n_mtp = 0
    # Disable MTP aux loss during QA fine-tune (answer CE only).
    config.mtp_loss_weight = 0.0

    model = CharGPT(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    if tokenizer.vocab_size > model.config.vocab_size:
        old_vs = model.config.vocab_size
        model.expand_vocab(tokenizer.vocab_size)
        print(f"Expanded embeddings {old_vs} → {tokenizer.vocab_size}")

    print(
        f"Init from {ckpt_path} (step={ckpt.get('step')}, val={ckpt.get('val_loss')})"
    )
    print(f"Model params: {model.count_parameters():,}")

    # Encode each example separately for masked batches.
    examples: list[tuple[list[int], int]] = []
    for obj in pairs:
        text = format_example(obj)
        ids, ignore_until = encode_with_answer_mask(tokenizer, text)
        if args.full_loss:
            ignore_until = 0
        if len(ids) >= 2:
            examples.append((ids, ignore_until))

    n = len(examples)
    split = max(1, int(n * (1.0 - args.val_ratio)))
    # Keep at least one val example when possible.
    if split >= n and n > 1:
        split = n - 1
    train_examples = examples[:split]
    val_examples = examples[split:] if split < n else examples[:1]
    print(
        f"Examples train={len(train_examples)} val={len(val_examples)} | "
        f"block_size={config.block_size} | "
        f"loss={'full' if args.full_loss else 'answer-only'}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out_dir / "vocab.txt")
    best_val = math.inf
    t0 = time.time()
    t_last = t0
    block_size = config.block_size

    for step in range(1, args.max_iters + 1):
        xb, yb = make_masked_batch(
            train_examples, args.batch_size, block_size, device
        )
        logits, _ = model(xb)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.view(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if args.step_sleep > 0:
            time.sleep(args.step_sleep)

        if step % 25 == 0 or step == 1:
            now = time.time()
            print(
                f"step {step:5d}/{args.max_iters}  loss={loss.item():.4f}  "
                f"(+{now - t_last:.1f}s, total {now - t0:.1f}s)"
            )
            t_last = now

        if step % args.eval_interval == 0 or step == args.max_iters:
            val_loss = estimate_loss(
                model, val_examples, args.batch_size, block_size, device
            )
            now = time.time()
            print(
                f"eval @ {step}: val={val_loss:.4f}  "
                f"(+{now - t_last:.1f}s, total {now - t0:.1f}s)"
            )
            t_last = now
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.__dict__,
                "step": step,
                "val_loss": val_loss,
                "level": "word",
                "task": "qa_finetune",
                "data_file": str(args.data),
                "init_from": str(args.init_from),
                "answer_only": not args.full_loss,
            }
            torch.save(payload, args.out_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(payload, args.out_dir / "best.pt")
                print(f"  saved best.pt (val={best_val:.4f})")

    meta = {
        "task": "qa_finetune",
        "level": "word",
        "data_file": str(args.data),
        "n_pairs": len(pairs),
        "init_from": str(args.init_from),
        "out_dir": str(args.out_dir),
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "n_mtp": config.n_mtp,
        "max_iters": args.max_iters,
        "learning_rate": args.learning_rate,
        "answer_only": not args.full_loss,
        "best_val_loss": best_val,
        "device": str(device),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done. Fine-tuned checkpoint in {args.out_dir}")
    print(
        "Query with:\n"
        f'  .venv/bin/python complete_gpt_word.py --ckpt-dir {args.out_dir} '
        '--prompt "### Instruction:\\nAnswer the question using yoga knowledge '
        'from the books.\\n\\n### Question:\\nWhat is Kundalini?\\n\\n### Answer:\\n"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
