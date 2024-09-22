import argparse

import numpy as np
from boostsa import Bootstrap
from prompt_evaluation import load_dataset

np.random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob_path1", type=str)
    parser.add_argument("--prob_path2", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--n_loops", type=int, default=1000)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()


    ds = load_dataset(
        data_path=args.data_path,
        task=args.task,
        lang=args.lang,
        labels=False,
        split="test",
        mode="debug"
    )[0]
    labels = np.array(ds["label"])
    preds1 = np.argmax(np.load(args.prob_path1), axis=-1)
    preds2 = np.argmax(np.load(args.prob_path2), axis=-1)
    boot = Bootstrap(save_outcomes=False, dir_out="")
    results = boot.test(
        targs=labels,
        h0_preds=preds1,
        h1_preds=preds2,
        n_loops=args.n_loops,
        sample_size=0.5,
        verbose=True,
    )
    # print(results)
    is_significant = (
        True
        if results[0].loc["h1"]["s_acc"] == "*" or results[0].loc["h1"]["s_acc"] == "**"
        else False
    )
    print()
    print("Statistically significant difference:", is_significant)
