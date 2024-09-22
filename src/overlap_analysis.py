import argparse
import os
from typing import List

import pandas as pd
from visualize import plot_heatmap

LANG_MAPPER = {
    "en": "En",
    "fr": "Fr",
    "es": "Es",
    "zh": "Zh",
    "ru": "Ru",
    "hi": "Hi",
    "th": "Th",
    "sw": "Sw",
    "fr-en": "Fr-En",
    "es-en": "Es-En",
    "zh-en": "Zh-En",
    "ru-en": "Ru-En",
    "hi-en": "Hi-En",
    "th-en": "Th-En",
    "sw-en": "Sw-En",
    "en-fr": "En-Fr",
    "en-es": "En-Es",
    "en-zh": "En-Zh",
    "en-ru": "En-Ru",
    "en-hi": "En-Hi",
    "en-th": "En-Th",
    "en-sw": "En-Sw",
}


def overlap_analysis(
    root_dir: str, langs: List[str], save_dir: str, ratio: float = 0.3
):
    os.makedirs(save_dir, exist_ok=True)
    df_dict = {}
    to_en_langs = []
    from_en_langs = []
    for lang in langs:
        # if lang == "sw":
        #     df_dict[lang] = pd.read_csv(f"{root_dir}/th/hidden_states.csv")
        # else:
        df_dict[lang] = pd.read_csv(f"{root_dir}/{lang}/hidden_states.csv")
        if lang != "en":
            # if lang == "sw":
            #     to_en = lang + "-en"
            #     df_dict[to_en] = pd.read_csv(f"{root_dir}/th-en/hidden_states.csv")
            #     to_en_langs.append(to_en)
            #     from_en = "en-" + lang
            #     df_dict[from_en] = pd.read_csv(f"{root_dir}/en-th/hidden_states.csv")
            #     from_en_langs.append(from_en)
            # else:
            to_en = lang + "-en"
            df_dict[to_en] = pd.read_csv(f"{root_dir}/{to_en}/hidden_states.csv")
            to_en_langs.append(to_en)
            # from_en = "en-" + lang
            # df_dict[from_en] = pd.read_csv(f"{root_dir}/{from_en}/hidden_states.csv")
            # from_en_langs.append(from_en)

    num_layers = len(df_dict[langs[0]])
    for l in range(num_layers):
        overlap_df = pd.DataFrame()
        for lang in langs + to_en_langs + from_en_langs:
            lang_sorted_indexs = df_dict[lang].loc[l, "dim0":].to_numpy().argsort()
            num_indexs = round(len(lang_sorted_indexs) * 0.4)
            lang_highest_indexs = lang_sorted_indexs[-num_indexs:]
            for lang2 in langs + to_en_langs + from_en_langs:
                lang2_sorted_indexs = (
                    df_dict[lang2].loc[l, "dim0":].to_numpy().argsort()
                )
                lang2_lowest_indexs = lang2_sorted_indexs[:num_indexs]
                overlap_ratio = (
                    len(set(lang_highest_indexs) & set(lang2_lowest_indexs))
                    / num_indexs
                ) * 100
                overlap_df.loc[LANG_MAPPER[lang], LANG_MAPPER[lang2]] = overlap_ratio

        plot_heatmap(
            overlap_df,
            annot=True,
            vmin=0,
            vmax=30,
            save_path=f"{save_dir}/layer{l}.overlap.{ratio}.png",
            fmt=".0f",
            xy_labels=[
                f"Bottom-{int(ratio * 100)}% magnitude features",
                f"Top-{int(ratio * 100)}% magnitude features",
            ],
            colorbar_title="Overlap ratio (%)",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--langs", nargs="*", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--ratio", type=float, default=0.3)
    args = parser.parse_args()

    overlap_analysis(
        root_dir=args.root_dir,
        langs=args.langs,
        save_dir=args.save_dir,
        ratio=args.ratio,
    )