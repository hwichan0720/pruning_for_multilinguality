import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from data.load_data import get_loaders
from transformers import AutoModelForCausalLM, AutoTokenizer
from visualize import plot_bar


def analyze(
    model_name: str,
    device: str,
    task_name: str,
    save_dir: str,
    num_examples: int,
    num_shots: int,
    src_lang: str,
    tgt_lang: str = None,
):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    if "mGPT" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/hwichan/prune_for_multilingual/outputs/ai-forever/mGPT/tokenizer"
        )
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataloader = get_loaders(
        task_name,
        nsamples=num_examples,
        seed=0,
        shots=num_shots,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        tokenizer=tokenizer,
    )
    hidden_states_per_layer = defaultdict(lambda: torch.tensor(0))
    for i, example in enumerate(dataloader[0]):
        outs = model(
            input_ids=example[0].to(device),
            attention_mask=example[-2].to(device),
            output_hidden_states=True,
        )
        num_dimensions = outs.hidden_states[0].shape[-1]
        num_layers = len(outs.hidden_states)

        for l in range(num_layers):
            hidden_states = outs.hidden_states[l].detach().cpu()
            if len(hidden_states.shape) == 3:
                hidden_states = hidden_states.reshape((-1, hidden_states.shape[-1]))
            hidden_states = torch.norm(hidden_states, p=2, dim=0).numpy()
            if hidden_states_per_layer[l] == torch.tensor(0):
                hidden_states_per_layer[l] = hidden_states
                continue

            hidden_states_per_layer[l] += hidden_states

    df = pd.DataFrame(
        columns=["layer"] + [f"dim{str(i)}" for i in range(num_dimensions)],
    )
    for l, hidden_states in hidden_states_per_layer.items():
        hidden_states_per_layer[l] = hidden_states / num_examples
        df.loc[len(df)] = [l] + list(hidden_states_per_layer[l])
        plot_bar(
            x=[str(i) for i in hidden_states_per_layer[l].argsort()[::-1][:20]],
            y=np.sort(hidden_states_per_layer[l])[::-1][:20],
            save_path=f"{save_dir}/layer{l}.top{20}.png",
        )
        df.to_csv(f"{save_dir}/hidden_states.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, help="Model name (e.g. facebook/xglm-1.7B)"
    )
    parser.add_argument("--task_name", type=str, help="Task name (lm or mt)")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_examples", type=int)
    parser.add_argument("--num_shots", type=int)
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.task_name == "mt":
        save_dir = f"{args.save_dir}/{args.model_name}/analysis/{args.src_lang}-{args.tgt_lang}"
    elif args.task_name == "mono":
        save_dir = f"{args.save_dir}/{args.model_name}/analysis/{args.src_lang}"

    os.makedirs(save_dir, exist_ok=True)
    analyze(
        model_name=args.model_name,
        device=device,
        task_name=args.task_name,
        save_dir=save_dir,
        num_examples=args.num_examples,
        num_shots=args.num_shots,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )
