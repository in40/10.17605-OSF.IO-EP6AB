"""
make_scaling_figure.py — scaling bar chart (4B vs 9B), RU + EN variants.
Built from existing report numbers (no model re-run). DejaVu Sans for Cyrillic.
Run: .venv-docx/bin/python make_scaling_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

b4 = [0.895, 0.877, 0.480]
b9 = [0.907, 0.890, 0.459]
x = np.arange(3)
w = 0.36


def build(groups, title, tfidf_lbl, h1_lbl, chance_lbl, l4, l9, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    r4 = ax.bar(x - w/2, b4, w, label=l4, color="#2b6cb0", alpha=0.85)
    r9 = ax.bar(x + w/2, b9, w, label=l9, color="#63b3ed", alpha=0.85)
    ax.axhline(0.929, color="#d1495b", ls="--", lw=1.5)
    ax.text(2.5, 0.933, tfidf_lbl, color="#d1495b", fontsize=9, va="bottom", ha="right")
    ax.axhline(0.90, color="gray", ls=":", lw=1.2)
    ax.text(2.5, 0.885, h1_lbl, color="gray", fontsize=8, va="top", ha="right")
    ax.axhline(0.5, color="black", ls="-.", lw=1.0, alpha=0.6)
    ax.text(-0.45, 0.505, chance_lbl, color="black", fontsize=8, va="bottom", ha="left", alpha=0.7)
    for bars in (r4, r9):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", (rect.get_x()+rect.get_width()/2, h),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.0); ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.legend(loc="center", bbox_to_anchor=(0.37, 0.55), fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print("saved ->", out)


build(["Объединённый AUROC\n(дельта)", "H1 среднее\nпо семействам", "Согласованный H2\n(обёртка)"],
      "Масштаб модели 4B → 9B: дельта почти не меняется",
      "TF-IDF = 0.929", "порог H1 = 0.90", "случайное угадывание = 0.5",
      "Qwen3.5-4B", "Qwen3.5-9B", "fig_scaling_ru.png")

build(["Pooled AUROC\n(delta)", "H1 mean\nper-family", "Matched H2\n(wrapper)"],
      "Model scale 4B → 9B: the delta barely changes",
      "TF-IDF = 0.929", "H1 threshold = 0.90", "chance = 0.5",
      "Qwen3.5-4B", "Qwen3.5-9B", "fig_scaling_en.png")
