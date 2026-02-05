#!/usr/bin/env python3
"""
03_sft_train.py

Loads the tokenized dataset from save_to_disk and runs SFT as standard
causal language modeling fine-tuning using 🤗 Transformers Trainer.

Assumes dataset has: input_ids.
Labels are created automatically via DataCollatorForLanguageModeling(mlm=False).
"""

import argparse

import math

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default="data/flavor_texts_packed_512",
        help="Tokenized dataset directory (save_to_disk)",
    )
    p.add_argument(
        "--model", required=True, help="Base model checkpoint (HF name or local path)"
    )
    p.add_argument(
        "--out", default="runs/flavor_sft", help="Output dir for checkpoints"
    )
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument(
        "--eval-split",
        type=float,
        default=0.01,
        help="Holdout fraction for eval (0 disables)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help='Resume from checkpoint path, or "latest" to auto-detect the last checkpoint.',
    )
    args = p.parse_args()

    ds = load_from_disk(args.dataset)

    # Optional eval split
    if args.eval_split and args.eval_split > 0:
        split = ds.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # Good defaults for modern setups; Trainer will pick fp16/bf16 depending on hardware if you enable it.
    # We'll enable bf16 if supported, else fp16 if CUDA, else neither.
    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    fp16 = use_cuda and not bf16

    # Number of optimizer steps per epoch (after grad accumulation)
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    max_steps = math.ceil(steps_per_epoch * args.epochs)

    warmup_steps = int(args.warmup_ratio * max_steps)
    print(
        f"steps/epoch={steps_per_epoch}, max_steps={max_steps}, warmup_steps={warmup_steps}"
    )

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_total_limit=3,
        report_to="none",
        seed=args.seed,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    resume = args.resume
    if resume == "latest":
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved final model -> {args.out}")


if __name__ == "__main__":
    main()
