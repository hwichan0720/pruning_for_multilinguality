import argparse

import pandas as pd


def save(sents, path):
    with open(path, "w") as f:
        for s in sents:
            print(s, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str)
    parser.add_argument("--path2", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--get_ref", action="store_true")
    args = parser.parse_args()

    hyp1 = pd.read_table(args.path1)
    hyp2 = pd.read_table(args.path2)
    save(
        list(hyp1["premise"].values) + list(hyp1["hypothesis"].values),
        f"{args.output_dir}/hyp1.txt",
    )
    save(
        list(hyp2["premise"].values) + list(hyp2["hypothesis"].values),
        f"{args.output_dir}/hyp2.txt",
    )

    if args.get_ref:
        from datasets import load_dataset

        ref = load_dataset("xnli", "en", split="test")
        save(
            ref["premise"] + ref["hypothesis"],
            f"{args.output_dir}/ref.txt",
        )
