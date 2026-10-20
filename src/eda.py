"""EDA plots. Run: python -m src.eda"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.config import (CATEGORICAL_FEATURES, FIG_DIR, NUMERIC_FEATURES,  # noqa: E402
                        TARGET)
from src.data import load_clean  # noqa: E402

PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
LABELS = {0: "No disease", 1: "Disease"}


def _legend_handles():
    return [Patch(color=PALETTE[k], label=LABELS[k]) for k in (0, 1)]


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def class_balance(df):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v} ({v / len(df):.0%})",
                ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    return _save(fig, "class_balance.png")


def numeric_histograms(df):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=TARGET, palette=PALETTE, kde=True,
                     bins=25, ax=ax, element="step")
        ax.set_title(f"{col} by outcome")
        if ax.legend_:
            ax.legend_.remove()
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].legend(handles=_legend_handles(), loc="center", fontsize=12)
    fig.suptitle("Numeric feature distributions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "numeric_histograms.png")


def categorical_counts(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=TARGET, palette=PALETTE, ax=ax)
        ax.set_title(col)
        if ax.legend_:
            ax.legend_.remove()
    fig.legend(handles=_legend_handles(), loc="upper right")
    fig.suptitle("Categorical features vs outcome", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "categorical_counts.png")


def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    corr = df.corr(numeric_only=True)
    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    for i, r in enumerate(corr.index):
        for j, c in enumerate(corr.columns):
            mask.iloc[i, j] = j > i
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Correlation heatmap")
    return _save(fig, "correlation_heatmap.png")


def target_correlation(df):
    corr = df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(corr.index, corr.values,
            color=[PALETTE[1] if v > 0 else PALETTE[0] for v in corr.values])
    ax.set_title("Correlation with heart disease")
    ax.axvline(0, color="k", lw=.8)
    return _save(fig, "target_correlation.png")


def boxplots(df):
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, palette=PALETTE,
                    legend=False, ax=ax)
        ax.set_xticks([0, 1], ["No", "Yes"])
    fig.suptitle("Outliers / spread by outcome")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "boxplots.png")


def missing_values(df):
    miss = df.isna().sum()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(miss.index, miss.values, color="#264653")
    ax.set_title("Missing values per column")
    for i, v in enumerate(miss.values):
        ax.text(i, v + .05, str(v), ha="center")
    return _save(fig, "missing_values.png")


def run_all(df=None):
    sns.set_theme(style="whitegrid")
    df = load_clean() if df is None else df
    return [f(df) for f in (class_balance, numeric_histograms, categorical_counts,
                            correlation_heatmap, target_correlation, boxplots,
                            missing_values)]


if __name__ == "__main__":
    for p in run_all():
        print("saved", p)
