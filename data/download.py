import argparse
import os
from typing import List

import datasets
import numpy as np
from datasets import load_dataset


def to_csv(dataset: datasets.dataset_dict.DatasetDict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for split, data in dataset.items():
        data.to_csv(f"{save_dir}/{split}.csv", index=False)


def download_amazon_review(
    save_dir: str, langs: List[str] = ["en", "fr", "ja", "zh", "es", "de"]
) -> None:
    for lang in langs:
        dataset = load_dataset(f"SetFit/amazon_reviews_multi_{lang}")
        dataset = dataset.rename_column("text", "sentence")
        to_csv(dataset, f"{save_dir}/MARC/{lang}")

        # Constructing binary classification  dataset
        for split, data in dataset.items():
            labels = np.array(data["label"])
            data = data.select(np.where(labels != 2)[0])
            data = data.map(
                lambda batch: {"label": [1 if x > 2 else 0 for x in batch["label"]]},
                batched=True,
            )
            dataset[split] = data

        to_csv(dataset, f"{save_dir}/MARC-2/{lang}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    download_amazon_review(save_dir=args.save_dir)
