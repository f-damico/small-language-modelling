#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


ROOT = Path(__file__).resolve().parent

TARGET_VOCAB_SIZE = 8192

TRAIN_TXT = ROOT / "TinyStoriesV2-GPT4-train.txt"
VALID_TXT = ROOT / "TinyStoriesV2-GPT4-valid.txt"

TOKENIZER_JSON = ROOT / f"tinystories.BPE{TARGET_VOCAB_SIZE}.tokenizer.json"
META_JSON = ROOT / f"tinystories.BPE{TARGET_VOCAB_SIZE}.meta.json"

TRAIN_NPY = ROOT / f"tinystories.BPE{TARGET_VOCAB_SIZE}.train.npy"
VALID_NPY = ROOT / f"tinystories.BPE{TARGET_VOCAB_SIZE}.valid.npy"


def load_stories(path: Path) -> list[str]:
    print(f"[LOAD] {path}", flush=True)
    with path.open("r", encoding="utf-8") as f:
        data = f.read()

    print(f"[INFO] {len(data)} characters in {path.name}", flush=True)

    # This is exactly the split used in the repo notebook:
    # stories = data.split("<|endoftext|>")
    stories = data.split("<|endoftext|>")
    stories = [story.strip() for story in stories if story.strip()]

    print(f"[INFO] loaded {len(stories)} stories from {path.name}", flush=True)
    print("[INFO] first story:")
    print(stories[0][:1000])
    print()
    return stories


def train_tokenizer(train_stories: list[str], valid_stories: list[str]) -> Tokenizer:
    print("[TOKENIZER] training BPE8192 tokenizer exactly as notebook", flush=True)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=TARGET_VOCAB_SIZE,
        special_tokens=["[UNK]", "<eos>"],
    )

    # Exact notebook choice.
    tokenizer.pre_tokenizer = Whitespace()

    # Exact notebook logic: iterator contains train + valid stories.
    iterator = train_stories + valid_stories
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    print("[TOKENIZER] vocab size:", tokenizer.get_vocab_size(), flush=True)
    print("[TOKENIZER] eos id:", tokenizer.token_to_id("<eos>"), flush=True)
    print("[TOKENIZER] unk id:", tokenizer.token_to_id("[UNK]"), flush=True)

    tokenizer.save(str(TOKENIZER_JSON))
    print(f"[TOKENIZER] saved {TOKENIZER_JSON}", flush=True)
    return tokenizer


def tokenize_and_save(tokenizer: Tokenizer, stories: list[str], out_path: Path, eos_id: int) -> int:
    print(f"[TOKENIZE] writing {out_path}", flush=True)

    tokenised: list[int] = []
    lengths: list[int] = []

    for i, story in enumerate(stories):
        output = tokenizer.encode(story)
        ids = output.ids

        # Exact notebook logic:
        # tokenised += output.ids + [1]
        tokenised += ids + [eos_id]
        lengths.append(len(output))

        if (i + 1) % 100000 == 0:
            print(
                f"[TOKENIZE] {i+1:,}/{len(stories):,} stories, "
                f"{len(tokenised):,} tokens",
                flush=True,
            )

    arr = np.asarray(tokenised, dtype=np.int32)
    np.save(out_path, arr)

    print(f"[DONE] {out_path.name}", flush=True)
    print(f"       tokens: {len(arr):,}", flush=True)
    print(f"       min/max story length: {min(lengths)} / {max(lengths)}", flush=True)
    print(f"       average story length: {sum(lengths) / len(lengths):.4f}", flush=True)

    return int(len(arr))


def main() -> None:
    if not TRAIN_TXT.exists():
        raise FileNotFoundError(TRAIN_TXT)
    if not VALID_TXT.exists():
        raise FileNotFoundError(VALID_TXT)

    train_stories = load_stories(TRAIN_TXT)
    valid_stories = load_stories(VALID_TXT)

    tokenizer = train_tokenizer(train_stories, valid_stories)

    eos_id = tokenizer.token_to_id("<eos>")
    unk_id = tokenizer.token_to_id("[UNK]")

    if eos_id != 1:
        raise RuntimeError(f"Expected eos_id=1, got {eos_id}")
    if unk_id != 0:
        raise RuntimeError(f"Expected unk_id=0, got {unk_id}")

    train_size = tokenize_and_save(tokenizer, train_stories, TRAIN_NPY, eos_id)
    valid_size = tokenize_and_save(tokenizer, valid_stories, VALID_NPY, eos_id)

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_token_id": eos_id,
        "unk_token_id": unk_id,
        "tokenizer_file": f"tinystories/tinystories.BPE{TARGET_VOCAB_SIZE}.tokenizer.json",
        "train_size": train_size,
        "valid_size": valid_size,
    }

    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[META] saved {META_JSON}", flush=True)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
