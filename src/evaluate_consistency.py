import argparse
from typing import List

import numpy as np


def softmax(x: np.ndarray):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cal_p_at_j(pred1: np.ndarray, pred2: np.ndarray, j: int):
    intersections = set(pred1[:j]) & set(pred2[:j])
    return len(intersections) / j


def cal_rankc(preds1: np.ndarray, preds2: np.ndarray):
    num_examples, num_queries = preds1.shape
    weights = softmax(np.array([i for i in range(num_queries)]))[::-1]
    rankc = np.zeros(num_examples)
    for i in range(num_examples):
        for j in range(num_queries):
            p_at_j = cal_p_at_j(pred1=preds1[i], pred2=preds2[i], j=j + 1)
            rankc[i] += weights[j] * p_at_j
    return rankc.mean()


def consistence_analysis(
    root_dir: str,
    langs: List[str],
):
    en_probs = np.argsort(np.load(f"{root_dir}/en.test.npy"), axis=-1)
    rankc_per_lang = np.zeros(len(langs))
    for i, lang in enumerate(langs):
        lang_probs = np.argsort(np.load(f"{root_dir}/{lang}.test.npy"), axis=-1)
        rankc = cal_rankc(en_probs, lang_probs)
        print(f"en-{lang} rankc = ", rankc)
        rankc_per_lang[i] = rankc
    print("Averaged rankc across each languages = ", rankc_per_lang.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--langs", nargs="*", type=str)
    args = parser.parse_args()

    consistence_analysis(
        root_dir=args.root_dir,
        langs=args.langs,
    )
