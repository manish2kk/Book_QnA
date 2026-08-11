#!/usr/bin/env python3
"""Interactive / one-shot completion with a pretrained CharGPT (MTP speculative)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gpt.model import CharGPT, config_from_dict
from gpt.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT_DIR = ROOT / "gpt_checkpoints" / "kundalini_char_gpt"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_dir: Path, device: torch.device) -> tuple[CharGPT, CharTokenizer]:
    ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint in {ckpt_dir}. Run: .venv/bin/python train_gpt.py"
        )
    vocab_path = ckpt_dir / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab at {vocab_path}")

    tokenizer = CharTokenizer.load(vocab_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = config_from_dict(ckpt["config"])
    has_mtp = any(k.startswith("mtp_heads") and ".out.weight" in k for k in ckpt["model"])
    if not has_mtp:
        config.n_mtp = 0
    model = CharGPT(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    mode = f"MTP speculative (n_mtp={config.n_mtp})" if config.n_mtp > 0 else "autoregressive"
    print(
        f"Loaded {ckpt_path.name} (step={ckpt.get('step')}, val={ckpt.get('val_loss')}, {mode})"
    )
    return model, tokenizer


@torch.no_grad()
def complete(
    model: CharGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 400,
    temperature: float = 0.8,
    top_k: int = 40,
    speculative: bool | None = None,
) -> str:
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = tokenizer.encode(" ")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        speculative=speculative,
    )
    return tokenizer.decode(out[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--prompt", type=str, default=None, help="One-shot prompt; omit for REPL")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--speculative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use MTP speculative decoding when the checkpoint has MTP heads "
        "(default: on if n_mtp>0). --no-speculative forces plain AR.",
    )
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_model(args.ckpt_dir, device)

    if args.prompt is not None:
        print(
            complete(
                model,
                tokenizer,
                args.prompt,
                device,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                speculative=args.speculative,
            )
        )
        return 0

    print("CharGPT completion. Type a prompt and press Enter. Empty line / 'quit' to exit.")
    while True:
        try:
            prompt = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt or prompt.strip().lower() in {"quit", "exit", "q"}:
            break
        text = complete(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            speculative=args.speculative,
        )
        print("\n" + text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
