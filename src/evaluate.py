import argparse
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from data.preprocess import (load_create_context_function,
                             load_create_prompt_function_per_label)
from datasets import load_dataset as _load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def torch_fix_seed(seed: int = 42) -> None:
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def evaluate_clm(
    dataset: Dataset,
    model,
    tokenizer: AutoTokenizer,
    label_names: List[str],
    device: str,
    batch_size: int = 32,
):
    length = len(dataset)
    # all_probs = np.zeros((length * len(label_names)))
    all_preds = np.zeros(length)
    all_probs = np.zeros((length, len(label_names)))
    all_labels = np.zeros(length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    bar = tqdm(total=len(dataloader))

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            all_labels[idx * batch_size : (idx + 1) * batch_size] = batch[
                "label"
            ].numpy()
            probs = np.zeros((batch["label"].shape[0], len(label_names)))
            for i, label_name in enumerate(label_names):
                # print(batch[label_name])
                tokenized_inputs = tokenizer(
                    batch[label_name],
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = tokenized_inputs["input_ids"].to(device)
                masks = tokenized_inputs["attention_mask"].to(device)

                logits = model(
                    input_ids=input_ids, attention_mask=masks, labels=input_ids
                ).logits

                logits = logits.detach().cpu()  # batch_size * max_length * vocab_num
                input_ids = input_ids.detach().cpu()  # batch_size * max_length
                masks = masks.detach().cpu()  # batch_size * max_length

                # get target token logprobs for each time step
                logprobs = torch.gather(
                    F.log_softmax(logits, dim=2), 2, input_ids[:, 1:].unsqueeze(2)
                )
                # ignore the logprobs of time steps when mask token
                logprobs = logprobs * masks[:, 1:].unsqueeze(dim=2).numpy()
                # get average of per-token log probabilities
                probs[:, i] = logprobs.sum(axis=1).squeeze() / masks.numpy().sum(axis=1)

            preds = np.argmax(probs, axis=-1)
            all_preds[idx * batch_size : (idx + 1) * batch_size] = preds
            all_probs[idx * batch_size : (idx + 1) * batch_size] = probs
            bar.update(1)

    del input_ids, masks, logits, batch, tokenized_inputs
    print(compute_metrics(all_preds, all_labels))
    return compute_metrics(all_preds, all_labels), all_probs


def load_dataset(
    data_path: str,
    task: str,
    lang: str,
    labels: List[str],
    split: str = "test",
    mode: str = "test",
):
    if data_path:
        dataset = _load_dataset(
            "csv",
            data_files={
                split: f"{data_path}/{split}.csv",
            },
        )[split]
    else:
        dataset = _load_dataset(task, lang, split=split)
        if task == "paws-x":
            labels = ["paraphrase", "non_paraphrase"]
        else:
            labels = dataset.features["label"].names

    if mode == "debug":
        dataset = dataset.shuffle(seed=0).select(range(1000))
    return dataset, labels


def evaluate_pt(
    model,
    tokenizer,
    testset: Dataset,
    task: str,
    target_template_lang: str,
    batch_size: int,
    save_path: str,
):
    # prompt_function = load_create_prompt_function(task)
    prompt_function_per_label = load_create_prompt_function_per_label(task)

    testset = testset.map(
        lambda examples: prompt_function_per_label(
            examples=examples,
            contexts=[],
            template_lang=target_template_lang,
            tokenizer=tokenizer,
        )
    )
    scores, preds = evaluate_clm(
        dataset=testset.with_format("torch"),
        model=model,
        tokenizer=tokenizer,
        label_names=labels,
        device=device,
        batch_size=batch_size,
    )
    print(scores)
    np.save(save_path, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, help="Model name (e.g. facebook/xglm-1.7B)"
    )
    parser.add_argument("--task", type=str, help="Task name (e.g. xnli)")
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--labels", nargs="*", type=str)
    parser.add_argument("--target_lang", default="en", type=str)
    parser.add_argument("--target_template_lang", default="en", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    # load dataset
    testset, labels = load_dataset(
        data_path=args.data_path,
        task=args.task,
        lang=args.target_lang,
        labels=args.labels,
        split=args.split,
        mode=args.mode,
    )

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    print(f"TGT: {args.target_lang}", f"TGT template: {args.target_template_lang}")
    torch_fix_seed(args.seed)

    evaluate_pt(
        model=model,
        tokenizer=tokenizer,
        testset=testset,
        task=args.task,
        target_template_lang=args.target_template_lang,
        batch_size=args.batch_size,
        save_path=args.save_path,
    )
