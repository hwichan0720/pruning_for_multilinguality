import argparse

import numpy as np
import pandas as pd
from visualize import plot_bar


def magnitude_analysis(root_dir: str, top_n: int):
    df = pd.read_csv(f"{root_dir}/hidden_states.csv")
    num_layers = len(df)
    for l in range(num_layers):
        features = df.loc[l, "dim0":].to_numpy()
        plot_bar(
            x=[str(i) for i in features.argsort()[::-1][:top_n]],
            y=np.sort(features)[::-1][:top_n],
            save_path=f"{root_dir}/layer{l}.top{top_n}.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    magnitude_analysis(root_dir=args.root_dir, top_n=args.top_n)
