import argparse
import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from lid.constants import DEFAULT_MODEL, DEFAULT_DATASET, DEFAULT_DATASET_FILE
from lid.data import load_lid_dataset, build_training_sample


class LIDDataset(Dataset):
    def __init__(self, texts, iso_codes, tokenizer, max_length=512):
        self.samples = [
            build_training_sample(t, c) for t, c in zip(texts, iso_codes)
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a causal LM for LID")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--dataset-file", default=DEFAULT_DATASET_FILE)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--sample-frac", type=float, default=0.1)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype
    )

    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model.to(device)

    print("Loading dataset...")
    df = load_lid_dataset(
        dataset_name=args.dataset,
        file_path=args.dataset_file,
        token=args.hf_token,
        sample_frac=args.sample_frac,
        random_state=args.seed,
    )

    val_size = int(len(df) * args.val_split)
    val_df = df.iloc[:val_size].reset_index(drop=True)
    train_df = df.iloc[val_size:].reset_index(drop=True)

    train_ds = LIDDataset(
        train_df["text"].tolist(),
        train_df["ISO-693-3"].tolist(),
        tokenizer,
        args.max_length,
    )
    val_ds = LIDDataset(
        val_df["text"].tolist(),
        val_df["ISO-693-3"].tolist(),
        tokenizer,
        args.max_length,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, args.warmup_steps, total_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum
            pbar.set_postfix(loss=f"{total_loss / (step + 1):.4f}")

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    outputs = model(**batch)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.output_dir, "best")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  Saved best model to {save_path}")

    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
