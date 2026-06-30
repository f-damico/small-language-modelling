#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent

TOKENIZER_JSON = ROOT / "tinystories.BPE8192.tokenizer.json"
META_JSON = ROOT / "tinystories.BPE8192.meta.json"

TRAIN_TXT = ROOT / "TinyStoriesV2-GPT4-train.txt"
VALID_TXT = ROOT / "TinyStoriesV2-GPT4-valid.txt"

TRAIN_NPY = ROOT / "tinystories.BPE8192.train.npy"
VALID_NPY = ROOT / "tinystories.BPE8192.valid.npy"

DELIM = "<|endoftext|>"
CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB


def iter_stories_stream(path: Path):
    """
    Stream stories separated by <|endoftext|> without reading the full file.

    This reproduces:
        stories = data.split("<|endoftext|>")
        story = story.strip()
        if story: keep it
    """
    carry = ""

    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break

            text = carry + chunk
            parts = text.split(DELIM)

            # Last part may be incomplete, keep it for the next chunk.
            carry = parts.pop()

            for story in parts:
                story = story.strip()
                if story:
                    yield story

    carry = carry.strip()
    if carry:
        yield carry


def count_tokens(tokenizer: Tokenizer, txt_path: Path, eos_id: int) -> tuple[int, int]:
    total_tokens = 0
    total_stories = 0

    for story in iter_stories_stream(txt_path):
        ids = tokenizer.encode(story).ids
        total_tokens += len(ids) + 1  # + EOS
        total_stories += 1

        if total_stories % 100000 == 0:
            print(
                f"[COUNT] {txt_path.name}: "
                f"stories={total_stories:,} tokens={total_tokens:,}",
                flush=True,
            )

    print(
        f"[COUNT DONE] {txt_path.name}: "
        f"stories={total_stories:,} tokens={total_tokens:,}",
        flush=True,
    )
    return total_tokens, total_stories


def write_tokens(
    tokenizer: Tokenizer,
    txt_path: Path,
    out_path: Path,
    eos_id: int,
    expected_tokens: int,
) -> tuple[int, int]:
    print(f"[WRITE] {out_path.name}: allocating {expected_tokens:,} int32 tokens", flush=True)

    arr = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.int32,
        shape=(expected_tokens,),
    )

    pos = 0
    total_stories = 0

    for story in iter_stories_stream(txt_path):
        ids = tokenizer.encode(story).ids
        n = len(ids)

        if pos + n + 1 > expected_tokens:
            raise RuntimeError(
                f"Token count exceeded expected size for {txt_path.name}. "
                f"pos={pos}, n={n}, expected={expected_tokens}"
            )

        arr[pos : pos + n] = np.asarray(ids, dtype=np.int32)
        pos += n

        arr[pos] = eos_id
        pos += 1

        total_stories += 1

        if total_stories % 100000 == 0:
            print(
                f"[WRITE] {txt_path.name}: "
                f"stories={total_stories:,} tokens={pos:,}/{expected_tokens:,}",
                flush=True,
            )

    if pos != expected_tokens:
        raise RuntimeError(
            f"Final token count mismatch for {txt_path.name}: "
            f"wrote {pos:,}, expected {expected_tokens:,}"
        )

    arr.flush()

    print(
        f"[WRITE DONE] {out_path.name}: "
        f"stories={total_stories:,} tokens={pos:,}",
        flush=True,
    )
    return pos, total_stories


def main() -> None:
    if not TOKENIZER_JSON.exists():
        raise FileNotFoundError(TOKENIZER_JSON)
    if not META_JSON.exists():
        raise FileNotFoundError(META_JSON)
    if not TRAIN_TXT.exists():
        raise FileNotFoundError(TRAIN_TXT)
    if not VALID_TXT.exists():
        raise FileNotFoundError(VALID_TXT)

    with META_JSON.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))

    eos_id = int(meta.get("eos_token_id", 1))
    unk_id = int(meta.get("unk_token_id", 0))
    vocab_size = int(meta["vocab_size"])

    print("[INFO] tokenizer:", TOKENIZER_JSON)
    print("[INFO] vocab_size:", vocab_size)
    print("[INFO] eos_id:", eos_id)
    print("[INFO] unk_id:", unk_id)
    print("[INFO] expected train_size:", meta["train_size"])
    print("[INFO] expected valid_size:", meta["valid_size"])

    if tokenizer.get_vocab_size() != vocab_size:
        raise RuntimeError(
            f"Tokenizer vocab size mismatch: tokenizer={tokenizer.get_vocab_size()}, meta={vocab_size}"
        )

    if tokenizer.token_to_id("<eos>") != eos_id:
        raise RuntimeError(
            f"EOS mismatch: tokenizer={tokenizer.token_to_id('<eos>')}, meta={eos_id}"
        )

    if tokenizer.token_to_id("[UNK]") != unk_id:
        raise RuntimeError(
            f"UNK mismatch: tokenizer={tokenizer.token_to_id('[UNK]')}, meta={unk_id}"
        )

    # First pass: count and verify exact sizes.
    train_count, train_stories = count_tokens(tokenizer, TRAIN_TXT, eos_id)
    valid_count, valid_stories = count_tokens(tokenizer, VALID_TXT, eos_id)

    if train_count != int(meta["train_size"]):
        print(
            f"[WARNING] Train size mismatch: counted {train_count:,}, "
            f"old meta says {meta['train_size']:,}. "
            "Using counted size.",
            flush=True,
        )

    if valid_count != int(meta["valid_size"]):
        print(
            f"[WARNING] Valid size mismatch: counted {valid_count:,}, "
            f"old meta says {meta['valid_size']:,}. "
            "Using counted size.",
            flush=True,
        )

    # Update metadata to match the actual .txt files being tokenized.
    meta["train_size"] = int(train_count)
    meta["valid_size"] = int(valid_count)

    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[META] updated metadata with counted token sizes", flush=True)

    # Second pass: write memmapped .npy files.
    write_tokens(tokenizer, TRAIN_TXT, TRAIN_NPY, eos_id, train_count)
    write_tokens(tokenizer, VALID_TXT, VALID_NPY, eos_id, valid_count)

    print("[DONE]")
    print("train npy:", TRAIN_NPY)
    print("valid npy:", VALID_NPY)
    print("train stories:", train_stories)
    print("valid stories:", valid_stories)


if __name__ == "__main__":
    main()
