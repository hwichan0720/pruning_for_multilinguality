from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["figure.autolayout"] = True


def plot_bar(
    x: np.ndarray,
    y: np.ndarray,
    save_path: str,
    title: str = None,
):
    plt.figure()
    sns.set_theme(font_scale=0.9)
    sns.barplot(x=[str(i) for i in x], y=y, log_scale=False)
    if len(x) > 20:
        plt.tick_params(
            labelbottom=False, labelright=False, labeltop=False, bottom=False
        )
    if title:
        plt.title(title)
    plt.xticks(rotation=45)
    plt.xlabel("Dimension")
    plt.ylabel("Magnitude (log scale)")
    plt.yscale("log")
    plt.ylim(10, 100000)
    plt.savefig(save_path)


def plot_bar_for_overlap(
    df: pd.DataFrame,
    save_path: str,
):
    plt.figure(figsize=(12, 4))
    sns.set_theme(font_scale=0.9)
    sns.barplot(
        x="Layer", y="Averaged Overlap Ratio", hue="Quadrant", data=df, errorbar=None
    )

    # if len(x) > 20:
    #     plt.tick_params(
    #         labelbottom=False, labelright=False, labeltop=False, bottom=False
    #     )
    # if title:
    #     plt.title(title)
    # plt.xticks(rotation=45)
    # plt.xlabel("Dimension")
    # plt.ylabel("Magnitude (log scale)")
    # plt.yscale("log")
    # plt.ylim(10, 100000)
    plt.savefig(save_path)


def plot_heatmap(
    array: np.ndarray,
    annot: bool = True,
    fontsize: int = 11.5,
    fmt: str = ".3f",
    save_path: str = None,
    vmin: float = None,
    vmax: float = None,
    title: str = None,
    xy_labels: List[str] = None,
    colorbar_title: str = None,
):
    plt.figure()
    sns.set_theme(font_scale=1.0)
    s = sns.heatmap(
        array,
        annot=annot,
        fmt=fmt,
        annot_kws={"fontsize": fontsize},
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": colorbar_title} if colorbar_title else None,
    )
    if xy_labels:
        s.set(xlabel=xy_labels[0], ylabel=xy_labels[1])
    plt.xticks(rotation=45)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
