import argparse
import os

import pandas as pd
import torch
from get_lang_vector import get_lang_vector
from models.bloom_lrp2 import LRP2BloomForCausalLM
from models.mgpt_lrp2 import LRP2mGPTForCausalLM
from models.xglm_lrp2 import LRP2XGLMForCausalLM
from transformers import AutoTokenizer

from prune_for_multilingual.src.evaluate_lrp2 import (evaluate_pt,
                                                      load_dataset,
                                                      torch_fix_seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, help="Model name (e.g. facebook/xglm-1.7B)"
    )
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_examples", type=int)
    parser.add_argument("--task", type=str, help="Task name (e.g. xnli)")
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--target_lang", default="en", type=str)
    parser.add_argument("--source_lang", default="en", type=str)
    parser.add_argument("--target_template_lang", default="en", type=str)
    parser.add_argument("--source_template_lang", default="en", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mode", type=str, default="test")
    args = parser.parse_args()

    vec_save_dir = f"{args.save_dir}/{args.model_name}/lang_vector"
    en_vec_path = f"{vec_save_dir}/en_vector.npy"
    tgt_lang_vec_path = f"{vec_save_dir}/{args.target_lang}_vector.npy"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(vec_save_dir, exist_ok=True)
    if not os.path.isfile(en_vec_path):
        get_lang_vector(
            model_name=args.model_name,
            device=device,
            save_dir=vec_save_dir,
            num_examples=args.num_examples,
            lang="en",
        )
    if not os.path.isfile(tgt_lang_vec_path):
        get_lang_vector(
            model_name=args.model_name,
            device=device,
            save_dir=vec_save_dir,
            num_examples=args.num_examples,
            lang=args.target_lang,
        )

    # load dataset
    testset, labels = load_dataset(
        data_path=None,
        task=args.task,
        lang=args.target_lang,
        labels=None,
        split=args.split,
        mode=args.mode,
    )

    do_hard = True
    if "xglm" in args.model_name:
        model = LRP2XGLMForCausalLM.from_pretrained(args.model_name).to(device)
        num_layers = model.config.num_layers
    elif "mGPT" in args.model_name:
        model = LRP2mGPTForCausalLM.from_pretrained(args.model_name).to(device)
        num_layers = model.config.n_layer
    elif "bloom" in args.model_name:
        model = LRP2BloomForCausalLM.from_pretrained(args.model_name).to(device)
        num_layers = model.config.n_layer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    print(f"TGT: {args.target_lang}", f"TGT template: {args.target_template_lang}")
    torch_fix_seed(args.seed)

    pred_save_dir = f"{args.save_dir}/{args.model_name}/{args.task}/lrp2"
    os.makedirs(pred_save_dir, exist_ok=True)
    pred_save_path = pred_save_dir
    result_df = pd.DataFrame(columns=["lirp", "lsrp", "accuracy"])

    best_config = {"lirp_layer": 0, "lsrp_layer": 0, "acc": 0}
    for lirp_layer in range(num_layers + 1):
        # if lirp_layer == 2:
        #     break
        for n in range(1, num_layers + 1 - lirp_layer):
            # for n in range(1, 3):
            lsrp_layer = lirp_layer + n
            print(f"LIRP: {lirp_layer}, LSRP: {lsrp_layer}")
            scores = evaluate_pt(
                model=model,
                tokenizer=tokenizer,
                testset=testset,
                task=args.task,
                data_path=None,
                select_method=None,
                seed=args.seed,
                n_shot=0,
                source_lang="en",
                source_template_lang=args.source_template_lang,
                target_template_lang=args.target_template_lang,
                prompt_type="hard",
                batch_size=args.batch_size,
                save_path=pred_save_path,
                src_vec_path=en_vec_path,
                tgt_vec_path=tgt_lang_vec_path,
                lirp_layer=lirp_layer,
                lsrp_layer=lsrp_layer,
                labels=labels,
                device=device,
            )
            result_df.loc[len(result_df)] = [lirp_layer, lsrp_layer, scores["accuracy"]]
            result_df.to_csv(
                f"{pred_save_path}/lrp2.{args.target_lang}.csv", index=False
            )

            if best_config["acc"] < scores["accuracy"]:
                best_config["lirp_layer"] = lirp_layer
                best_config["lsrp_layer"] = lsrp_layer
                best_config["acc"] = scores["accuracy"]

    print()
    print("Best Configuration")
    print(best_config)
    print()
    print("Evaluate with Best Configuration")
    # load dataset
    testset, labels = load_dataset(
        data_path=None,
        task=args.task,
        lang=args.target_lang,
        labels=None,
        split="test",
        mode="test",
    )
    scores = evaluate_pt(
        model=model,
        tokenizer=tokenizer,
        testset=testset,
        task=args.task,
        data_path=None,
        select_method=None,
        seed=args.seed,
        n_shot=0,
        source_lang="en",
        source_template_lang=args.source_template_lang,
        target_template_lang=args.target_template_lang,
        prompt_type="hard",
        batch_size=args.batch_size,
        save_path=pred_save_path,
        src_vec_path=en_vec_path,
        tgt_vec_path=tgt_lang_vec_path,
        lirp_layer=best_config["lirp_layer"],
        lsrp_layer=best_config["lsrp_layer"],
        labels=labels,
        device=device,
    )
