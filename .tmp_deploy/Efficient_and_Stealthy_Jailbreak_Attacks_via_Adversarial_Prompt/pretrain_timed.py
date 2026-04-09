#!/usr/bin/env python
import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
)


def load_json_field(folder: Path, field: str | None = None):
    data = []
    for file in sorted(folder.glob("*.json")):
        with open(file, "r", encoding="utf-8") as f:
            content = json.load(f)
        if field:
            if isinstance(content, list):
                data.extend([item[field] for item in content if isinstance(item, dict) and field in item])
            elif isinstance(content, dict) and field in content:
                data.append(content[field])
        else:
            if isinstance(content, list):
                data.extend(content)
            elif isinstance(content, str):
                data.append(content)
    return data


def combine_template_instruction(template: str, instruction: str) -> str:
    return f"{template.strip()}\n\n{instruction.strip()}"


def create_causal_data(text: str, tokenizer, max_length: int = 512):
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class PromptRandomSampleDataset(Dataset):
    def __init__(self, templates, instructions, tokenizer, max_length: int = 512):
        self.prompts = [combine_template_instruction(t, i) for t in templates for i in instructions]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return create_causal_data(self.prompts[idx], self.tokenizer, max_length=self.max_length)


def parse_args():
    p = argparse.ArgumentParser(description="Timed pre-training runner")
    p.add_argument("--template-dir", required=True)
    p.add_argument("--prompt-dir", required=True)
    p.add_argument("--save-path", required=True)
    p.add_argument("--model-path", default="")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--subset-ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--random-init-if-load-fail", action="store_true")
    p.add_argument("--random-hidden-size", type=int, default=128)
    p.add_argument("--random-num-layers", type=int, default=2)
    p.add_argument("--random-num-heads", type=int, default=4)
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_random_llama(vocab_size: int, hidden_size: int, num_layers: int, num_heads: int):
    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=1024,
    )
    # Transformers >=4.52 may expect these plans to be iterable during post_init.
    if getattr(cfg, "base_model_tp_plan", None) is None:
        cfg.base_model_tp_plan = {}
    if getattr(cfg, "base_model_pp_plan", None) is None:
        cfg.base_model_pp_plan = {}
    return LlamaForCausalLM(cfg)


def build_random_gpt2(vocab_size: int, hidden_size: int, num_layers: int, num_heads: int):
    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_embd=hidden_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_positions=1024,
        n_ctx=1024,
        bos_token_id=1,
        eos_token_id=2,
    )
    return GPT2LMHeadModel(cfg)


def pick_lora_targets(model) -> list[str]:
    names = [n for n, _ in model.named_modules()]
    if any(n.endswith("q_proj") for n in names) and any(n.endswith("v_proj") for n in names):
        return ["q_proj", "v_proj"]
    if any("c_attn" in n for n in names):
        return ["c_attn"]
    if any("query_key_value" in n for n in names):
        return ["query_key_value"]
    return []


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    timings = {}
    t0 = time.perf_counter()

    t_data0 = time.perf_counter()
    templates = load_json_field(Path(args.template_dir), field=None)
    instructions = load_json_field(Path(args.prompt_dir), field="instruction")
    timings["load_data_sec"] = round(time.perf_counter() - t_data0, 4)

    if len(templates) == 0:
        raise RuntimeError(f"No templates found in {args.template_dir}")
    if len(instructions) == 0:
        raise RuntimeError(f"No instructions found in {args.prompt_dir}")

    t_tok0 = time.perf_counter()
    model_source = args.model_path.strip() if args.model_path.strip() else "Pretrained-Llama"
    tokenizer = None
    try:
        tok = AutoTokenizer.from_pretrained(model_source, use_fast=False)
        # Some checkpoints may incorrectly resolve to bool instead of tokenizer object.
        if isinstance(tok, bool):
            raise TypeError("AutoTokenizer returned bool.")
        tokenizer = tok
    except Exception:
        try:
            tokenizer = LlamaTokenizerFast.from_pretrained(model_source)
        except Exception:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(Path(model_source) / "tokenizer.json"))
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    timings["load_tokenizer_sec"] = round(time.perf_counter() - t_tok0, 4)

    model_load_error = None
    t_model0 = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_source)
        timings["model_init_mode"] = "from_pretrained"
    except Exception as e:
        model_load_error = str(e)
        if not args.random_init_if_load_fail:
            raise
        try:
            model = build_random_llama(
                vocab_size=max(tokenizer.vocab_size, len(tokenizer)),
                hidden_size=args.random_hidden_size,
                num_layers=args.random_num_layers,
                num_heads=args.random_num_heads,
            )
            timings["model_init_mode"] = "random_llama_fallback"
        except Exception:
            model = build_random_gpt2(
                vocab_size=max(tokenizer.vocab_size, len(tokenizer)),
                hidden_size=args.random_hidden_size,
                num_layers=args.random_num_layers,
                num_heads=args.random_num_heads,
            )
            timings["model_init_mode"] = "random_gpt2_fallback"
    timings["load_model_sec"] = round(time.perf_counter() - t_model0, 4)

    t_lora0 = time.perf_counter()
    lora_targets = pick_lora_targets(model)
    timings["lora_targets"] = lora_targets
    if lora_targets:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=lora_targets,
        )
        model = get_peft_model(model, peft_config)
    else:
        timings["lora_skipped"] = True
    timings["wrap_lora_sec"] = round(time.perf_counter() - t_lora0, 4)

    t_ds0 = time.perf_counter()
    dataset = PromptRandomSampleDataset(templates, instructions, tokenizer, max_length=args.max_length)
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    keep_n = max(1, int(len(all_indices) * args.subset_ratio))
    dataset = Subset(dataset, all_indices[:keep_n])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    timings["build_dataset_sec"] = round(time.perf_counter() - t_ds0, 4)

    device = resolve_device(args.device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    model.train()
    epoch_timings = []
    epoch_losses = []

    t_train0 = time.perf_counter()
    for epoch in range(args.epochs):
        ep0 = time.perf_counter()
        running_loss = 0.0
        steps = 0
        for batch in tqdm(dataloader, desc=f"Training epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += float(loss.item())
            steps += 1

        ep_time = time.perf_counter() - ep0
        epoch_timings.append(round(ep_time, 4))
        epoch_losses.append(round(running_loss / max(steps, 1), 6))

    timings["train_total_sec"] = round(time.perf_counter() - t_train0, 4)

    t_save0 = time.perf_counter()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    timings["save_sec"] = round(time.perf_counter() - t_save0, 4)

    timings["total_sec"] = round(time.perf_counter() - t0, 4)
    timings["num_templates"] = len(templates)
    timings["num_instructions"] = len(instructions)
    timings["num_samples"] = len(dataset)
    timings["batch_size"] = args.batch_size
    timings["epochs"] = args.epochs
    timings["device"] = str(device)
    timings["epoch_times_sec"] = epoch_timings
    timings["epoch_avg_loss"] = epoch_losses
    if model_load_error:
        timings["model_load_error"] = model_load_error[:1000]

    with open(save_path / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timings, f, ensure_ascii=False, indent=2)

    print("=== Pre-training Timing Summary ===")
    for k in [
        "model_init_mode",
        "load_data_sec",
        "load_tokenizer_sec",
        "load_model_sec",
        "wrap_lora_sec",
        "build_dataset_sec",
        "train_total_sec",
        "save_sec",
        "total_sec",
    ]:
        if k in timings:
            print(f"{k}: {timings[k]}")
    print("epoch_times_sec:", timings["epoch_times_sec"])
    print("epoch_avg_loss:", timings["epoch_avg_loss"])
    print(f"timing_file: {save_path / 'timing.json'}")


if __name__ == "__main__":
    main()
