import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from data.preprocess import (
    load_create_context_function,
    load_create_prompt_function_per_label,
)
from datasets import load_dataset as _load_dataset
from models.mgpt_lrp2 import LRP2mGPTForCausalLM
from models.xglm_lrp2 import LRP2XGLMForCausalLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


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


def construct_lang_vector(
    vecs: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    lirp_layer: int,
    lsrp_layer: int,
    pre_ranges: List[Tuple[int]],
    hyp_ranges: List[Tuple[int]],
    device: str,
):
    lang_vectors = {}
    lang_flags = torch.zeros(input_ids.shape)
    for i in range(len(pre_ranges)):
        pre_range = pre_ranges[i]
        hyp_range = hyp_ranges[i]
        mask = list(attention_mask[i].detach().cpu())
        start_index = mask.index(1)
        lang_flags[i][pre_range[0] + start_index : pre_range[1] + start_index] = (
            torch.ones(pre_range[1] - pre_range[0])
        )
        lang_flags[i][hyp_range[0] + start_index : hyp_range[1] + start_index] = (
            torch.ones(hyp_range[1] - hyp_range[0])
        )

    for layer in [lirp_layer, lsrp_layer]:
        lang_vector = torch.zeros(len(input_ids), len(input_ids[0]), vecs.shape[1]).to(
            vecs.dtype
        )
        lang_vector[torch.where(lang_flags == 1)] = vecs[layer]

        lang_vectors[layer] = lang_vector.to(device)

    return lang_vectors


def evaluate_clm(
    dataset: Dataset,
    model,
    tokenizer: AutoTokenizer,
    label_names: List[str],
    device: str,
    prompt_type: str = "hard",
    batch_size: int = 32,
    src_vec_path: str = None,
    tgt_vec_path: str = None,
    lirp_layer: int = None,
    lsrp_layer: int = None,
):
    length = len(dataset)
    all_preds = np.zeros(length)
    all_probs = np.zeros((length, len(label_names)))
    all_labels = np.zeros(length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    bar = tqdm(total=len(dataloader))

    # lang vector
    src_vecs = torch.from_numpy(np.load(src_vec_path))  # num_layer * dim
    tgt_vecs = torch.from_numpy(np.load(tgt_vec_path))  # num_layer * dim

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            all_labels[idx * batch_size : (idx + 1) * batch_size] = batch[
                "label"
            ].numpy()
            probs = np.zeros((batch["label"].shape[0], len(label_names)))
            for i, label_name in enumerate(label_names):
                tokenized_inputs = tokenizer(
                    batch[label_name],
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = tokenized_inputs["input_ids"].to(device)
                masks = tokenized_inputs["attention_mask"].to(device)

                if lirp_layer != None:
                    src_lang_vectors = construct_lang_vector(
                        vecs=src_vecs,
                        input_ids=input_ids,
                        attention_mask=masks,
                        lirp_layer=lirp_layer,
                        lsrp_layer=lsrp_layer,
                        pre_ranges=batch[f"{label_name}_pre_range"],
                        hyp_ranges=batch[f"{label_name}_hyp_range"],
                        device=device,
                    )
                    tgt_lang_vectors = construct_lang_vector(
                        vecs=tgt_vecs,
                        input_ids=input_ids,
                        attention_mask=masks,
                        lirp_layer=lirp_layer,
                        lsrp_layer=lsrp_layer,
                        pre_ranges=batch[f"{label_name}_pre_range"],
                        hyp_ranges=batch[f"{label_name}_hyp_range"],
                        device=device,
                    )
                else:
                    src_lang_vectors = None
                    tgt_lang_vectors = None

                if prompt_type == "hard":
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=masks,
                        labels=input_ids,
                        tgt_lang_vectors=tgt_lang_vectors,
                        src_lang_vectors=src_lang_vectors,
                        lirp_layer=lirp_layer,
                        lsrp_layer=lsrp_layer,
                    ).logits
                elif prompt_type == "soft":
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=masks,
                        labels=input_ids,
                        do_inference=True,
                    )
                    prompt_input_ids = outputs[1]
                    logits = outputs[0].logits
                    sent_num, _, vocab_num = logits.shape
                    logits = logits[torch.where(prompt_input_ids != -100)].view(
                        sent_num, -1, vocab_num
                    )
                    del prompt_input_ids, outputs

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
        labels = dataset.features["label"].names

    if mode == "debug":
        dataset = dataset.shuffle(seed=0).select(range(500))
    return dataset, labels


def evaluate_pt(
    model,
    tokenizer,
    testset: Dataset,
    task: str,
    data_path: str,
    select_method: str,
    seed: int,
    n_shot: int,
    source_lang: str,
    source_template_lang: str,
    target_template_lang: str,
    prompt_type: str,
    batch_size: int,
    save_path: str,
    labels: List[str],
    device: str,
    src_vec_path: str = None,
    tgt_vec_path: str = None,
    lirp_layer: int = None,
    lsrp_layer: int = None,
):
    context_function = load_create_context_function(task)
    # prompt_function = load_create_prompt_function(task)
    prompt_function_per_label = load_create_prompt_function_per_label(task)

    # create context examples
    contexts = []
    if n_shot > 0:
        trainset = _load_dataset(
            task=task,
            data_path=data_path,
            lang=source_lang,
            split="train",
        )

        if select_method == "random":
            trainset = trainset.shuffle(seed=seed).select(range(n_shot))
        elif select_method == "per_label":
            indexs = []
            for target_label in range(len(labels)):
                label_indexs = [
                    idx
                    for idx, label in enumerate(trainset["label"])
                    if target_label == label
                ]
                indexs += random.sample(label_indexs, n_shot)

            trainset = trainset.select(indexs).shuffle(seed=seed)
        contexts = [
            context_function(
                example=example,
                label_names=labels,
                template_lang=source_template_lang,
            )
            for example in trainset
        ]
        print(f"Number of context examples {len(contexts)}")

    testset = testset.map(
        lambda examples: prompt_function_per_label(
            examples=examples,
            contexts=contexts,
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
        prompt_type=prompt_type,
        batch_size=batch_size,
        src_vec_path=src_vec_path,
        tgt_vec_path=tgt_vec_path,
        lirp_layer=lirp_layer,
        lsrp_layer=lsrp_layer,
    )
    np.save(save_path, preds)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, help="Model name (e.g. facebook/xglm-1.7B)"
    )
    parser.add_argument("--task", type=str, help="Task name (e.g. xnli)")
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--labels", nargs="*", type=str)
    parser.add_argument("--prompt_type", default="hard", type=str, help="hard or soft")
    parser.add_argument("--target_lang", default="en", type=str)
    parser.add_argument("--source_lang", default="en", type=str)
    parser.add_argument("--target_template_lang", default="en", type=str)
    parser.add_argument("--source_template_lang", default="en", type=str)
    parser.add_argument("--n_shot", type=int, default=0)
    parser.add_argument(
        "--select_method",
        type=str,
        default="random",
        help="A method to select n_shot demonstrations (random or per_label)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--src_vec_path", type=str)
    parser.add_argument("--tgt_vec_path", type=str)
    parser.add_argument("--lirp_layer", type=int)
    parser.add_argument("--lsrp_layer", type=int)
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
    do_hard = True if args.prompt_type == "hard" else False
    if "xglm" in args.model_name:
        model = LRP2XGLMForCausalLM.from_pretrained(args.model_name).to(device)
    elif "mGPT" in args.model_name:
        model = LRP2mGPTForCausalLM.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    print(f"TGT: {args.target_lang}", f"TGT template: {args.target_template_lang}")
    torch_fix_seed(args.seed)

    scores = evaluate_pt(
        model=model,
        tokenizer=tokenizer,
        testset=testset,
        task=args.task,
        data_path=args.data_path,
        select_method=args.select_method,
        seed=args.seed,
        n_shot=args.n_shot,
        source_lang=args.source_lang,
        source_template_lang=args.source_template_lang,
        target_template_lang=args.target_template_lang,
        prompt_type=args.prompt_type,
        batch_size=args.batch_size,
        save_path=args.save_path,
        src_vec_path=args.src_vec_path,
        tgt_vec_path=args.tgt_vec_path,
        lirp_layer=args.lirp_layer,
        lsrp_layer=args.lsrp_layer,
        device=device,
        labels=labels,
    )
