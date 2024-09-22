import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from data.load_data import lang_codes
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_lang_vector(
    model_name: str,
    device: str,
    save_dir: str,
    num_examples: int,
    lang: str,
    seed: int = 0,
):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset(
        "facebook/flores",
        lang_codes[lang],
        split="dev",
    ).shuffle(seed=seed)

    hidden_states_per_layer = defaultdict(lambda: torch.tensor(0))
    for i in range(num_examples):
        sent = dataset[i]["sentence"]
        tokenized_inputs = tokenizer(sent, return_tensors="pt")
        outs = model(
            input_ids=tokenized_inputs["input_ids"].to(device),
            attention_mask=tokenized_inputs["attention_mask"].to(device),
            output_hidden_states=True,
        )
        num_layers = len(outs.hidden_states)

        for l in range(num_layers):
            hidden_states = outs.hidden_states[l].detach().cpu()
            if len(hidden_states.shape) == 3:
                hidden_states = hidden_states.reshape((-1, hidden_states.shape[-1]))
            hidden_states = torch.mean(hidden_states, axis=0).numpy()
            if hidden_states_per_layer[l] == torch.tensor(0):
                hidden_states_per_layer[l] = hidden_states
                continue

            hidden_states_per_layer[l] += hidden_states

    res = np.zeros((len(hidden_states_per_layer), len(hidden_states_per_layer[0])))
    for l, hidden_states in hidden_states_per_layer.items():
        res[l] = hidden_states / num_examples

    np.save(f"{save_dir}/{lang}_vector", res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, help="Model name (e.g. facebook/xglm-1.7B)"
    )
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_examples", type=int)
    parser.add_argument("--lang", type=str)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = f"{args.save_dir}/{args.model_name}/lang_vector"

    os.makedirs(save_dir, exist_ok=True)
    get_lang_vector(
        model_name=args.model_name,
        device=device,
        save_dir=save_dir,
        num_examples=args.num_examples,
        lang=args.lang,
    )
