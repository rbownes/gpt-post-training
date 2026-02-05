"""
02_tokenize_dataset.py

Loads the JSONL dataset with 🤗 Datasets, tokenizes with a HF tokenizer,
PACKS short examples into fixed-length blocks (better throughput for SFT),
and saves a tokenized Arrow dataset to disk (fast to reload for training).

Output dataset columns:
- input_ids
- attention_mask
- labels (same as input_ids for causal LM)
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--jsonl", default="data/flavor_texts.jsonl", help="Input JSONL path"
    )
    p.add_argument(
        "--model", required=True, help="HF model/tokenizer name or local path"
    )
    p.add_argument(
        "--out",
        default="data/flavor_texts_packed_512",
        help="Output dataset directory (save_to_disk)",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Packed sequence length. Typically match your training max_length (e.g. 256/512/1024).",
    )
    p.add_argument(
        "--num-proc", type=int, default=1, help="Parallelism for dataset.map"
    )
    p.add_argument(
        "--add-eos",
        action="store_true",
        help="Append EOS between examples before packing (recommended for GPT-style training).",
    )
    args = p.parse_args()

    jsonl_path = Path(args.jsonl)
    out_dir = Path(args.out)

    ds = load_dataset("json", data_files=str(jsonl_path))["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Common for GPT-style causal LMs
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(
            "Tokenizer has no eos_token_id; cannot pack reliably for causal LM."
        )

    def tok(batch):
        # No truncation here; we want full sequences before packing.
        enc = tokenizer(batch["text"], add_special_tokens=False)

        if args.add_eos:
            # Add EOS token between examples so the model sees clear boundaries.
            enc["input_ids"] = [ids + [eos_id] for ids in enc["input_ids"]]
            enc["attention_mask"] = [mask + [1] for mask in enc["attention_mask"]]

        return enc

    tok_ds = ds.map(
        tok,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    def pack(batch):
        # Concatenate across the batch
        input_ids = list(itertools.chain.from_iterable(batch["input_ids"]))
        attn = list(itertools.chain.from_iterable(batch["attention_mask"]))

        # Drop remainder to make exact blocks
        total_length = len(input_ids)
        total_length = (total_length // args.block_size) * args.block_size

        input_ids = input_ids[:total_length]
        attn = attn[:total_length]

        # Split into blocks
        input_blocks = [
            input_ids[i : i + args.block_size]
            for i in range(0, total_length, args.block_size)
        ]
        attn_blocks = [
            attn[i : i + args.block_size]
            for i in range(0, total_length, args.block_size)
        ]

        return {
            "input_ids": input_blocks,
            "attention_mask": attn_blocks,
            "labels": input_blocks.copy(),  # causal LM labels
        }

    packed_ds = tok_ds.map(
        pack,
        batched=True,
        num_proc=args.num_proc,
        desc=f"Packing to block_size={args.block_size}",
    )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    packed_ds.save_to_disk(str(out_dir))
    print(f"Saved packed tokenized dataset -> {out_dir}")


if __name__ == "__main__":
    main()
