import argparse

import torch
from data.load_data import get_loaders
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BloomForCausalLM, Conv1D, GPT2LMHeadModel)
from layerwrapper import WrappedGPT
from wanda.lib.prune import find_layers, return_given_alpha


def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if type(model) is BloomForCausalLM or type(model) is GPT2LMHeadModel:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]

        if type(model) is GPT2LMHeadModel:
            subset = find_layers(
                module=layer,
                layers=[Conv1D, torch.nn.Linear],
            )
        else:
            subset = find_layers(module=layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


def prepare_calibration_input(model, dataloader, device, args=None):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if type(model) is BloomForCausalLM or type(model) is GPT2LMHeadModel:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    if args:
        inps = torch.zeros(
            (args.nsamples, args.max_length, model.config.hidden_size),
            dtype=dtype,
            device=device,
        )
    else:
        inps = torch.zeros(
            (128, model.seqlength, model.config.hidden_size),
            dtype=dtype,
            device=device,
        )
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def prune_wanda_with_mt(
    args,
    model,
    tokenizer,
    data_type,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    drop_top_weights: bool = False,
):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    print("loading calibdation data")
    dataloader, data_types = get_loaders(
        data_type,
        nsamples=args.nsamples,
        seed=args.seed,
        shots=args.shots,
        tokenizer=tokenizer,
        src_lang=args.source_lang,
        tgt_lang=args.target_lang,
        max_length=args.max_length,
    )
    args.nsamples = len(dataloader)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args=args
        )

    if type(model) is BloomForCausalLM or type(model) is GPT2LMHeadModel:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        if type(model) is GPT2LMHeadModel:
            subset = find_layers(
                module=layer,
                layers=[Conv1D, torch.nn.Linear],
            )
        else:
            subset = find_layers(module=layer)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name], data_types=data_types)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if type(model) is BloomForCausalLM:
                    alibi = model.transformer.build_alibi_tensor(
                        dataloader[j][-2].to(device),
                        model.transformer.num_heads,
                        dtype=inps.dtype,
                    )
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        alibi=alibi,
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                    )[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            wrapped_layers[name].calculate_scaler_row()
            if type(subset[name]) is Conv1D:
                W_metric = torch.abs(subset[name].weight.data.T) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )
            else:
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  # initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(
                    W_metric,
                    dim=-1,
                    descending=drop_top_weights,
                    stable=True,
                )

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before
                    )
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (
                        alpha_hist[1] - alpha_hist[0] >= 0.001
                    ):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][
                        :, : int(W_metric.shape[1] * args.sparsity_ratio)
                    ]
                    W_mask.scatter_(1, indices, True)

            if type(subset[name]) is Conv1D:
                subset[name].weight.data[W_mask.T] = 0  # set weights to zero
            else:
                subset[name].weight.data[W_mask] = 0  # set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                if type(model) is BloomForCausalLM:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        alibi=alibi,
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        # position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_random(
    args,
    model,
):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if type(model) is BloomForCausalLM or type(model) is GPT2LMHeadModel:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        if type(model) is GPT2LMHeadModel:
            subset = find_layers(
                module=layer,
                layers=[Conv1D, torch.nn.Linear],
            )
        else:
            subset = find_layers(module=layer)

        for name in subset:
            print(f"pruning layer {i} name {name}")
            if type(subset[name]) is Conv1D:
                W_original = subset[name].weight.data.T
            else:
                W_original = subset[name].weight.data

            W_mask = (
                torch.rand(size=W_original.shape) < args.sparsity_ratio
            ).int() == 1  # initialize a mask to be all False

            assert (
                round(
                    torch.sum(W_mask).item()
                    / (W_original.shape[0] * W_original.shape[1]),
                    1,
                )
                == args.sparsity_ratio
            )
            if type(subset[name]) is Conv1D:
                subset[name].weight.data[W_mask.T] = 0  # set weights to zero
            else:
                subset[name].weight.data[W_mask] = 0  # set weights to zero

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/xglm-2.9B",
        help="Model name (e.g. facebook/xglm-1.7B)",
    )
    parser.add_argument("--data_type", type=str, default="mt")
    parser.add_argument(
        "--sparsity_ratio", type=float, default=0, help="Sparsity level"
    )
    parser.add_argument(
        "--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"]
    )
    parser.add_argument("--source_lang", default="zh", type=str)
    parser.add_argument("--target_lang", default="en", type=str)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--nsamples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_variant", action="store_true")
    parser.add_argument("--load_local_model", action="store_true")
    parser.add_argument(
        "--save_model", type=str, default=None, help="Path to save the pruned model."
    )
    parser.add_argument("--eval_task", type=str, default=None)
    parser.add_argument("--drop_top_weights", action="store_true")
    args = parser.parse_args()

    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert (
            args.sparsity_ratio == 0.5
        ), "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    if "mGPT" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/hwichan/prune_for_multilingual/outputs/ai-forever/mGPT/tokenizer"
        )
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.data_type == "random":
        prune_random(args=args, model=model)
    else:
        prune_wanda_with_mt(
            args=args,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prune_n=prune_n,
            prune_m=prune_m,
            data_type=args.data_type,
            drop_top_weights=args.drop_top_weights,
        )

    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)

    if args.save_model:
        if "mt" in args.data_type:
            save_dir = f"{args.save_model}/{args.model_name}/{args.source_lang}-{args.target_lang}/sparsity_type={args.sparsity_type}_ratio={args.sparsity_ratio}_shots={args.shots}_samples={args.nsamples}"
        elif "mono" in args.data_type:
            save_dir = f"{args.save_model}/{args.model_name}/{args.source_lang}/sparsity_type={args.sparsity_type}_ratio={args.sparsity_ratio}_shots={args.shots}_samples={args.nsamples}"
        elif args.data_type == "random":
            save_dir = f"{args.save_model}/{args.model_name}/random/sparsity_type={args.sparsity_type}_ratio={args.sparsity_ratio}"

        if args.drop_top_weights:
            save_dir += f"_top={args.drop_top_weights}"

        if "code" in args.data_type:
            save_dir += "_code"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)