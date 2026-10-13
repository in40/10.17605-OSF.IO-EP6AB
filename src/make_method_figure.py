"""
make_method_figure.py — two-panel method scheme (no model/data needed).
(a) simplified 5-block overview
(b) detailed two-pass delta pipeline as a single left-to-right chain
Run: .venv-docx/bin/python make_method_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 9})
C_IN = "#e8f0fe"; C_MODEL = "#dbe9d5"; C_OPS = "#fde8e8"; C_PROBE = "#fff3d6"; C_DEC = "#e5e0f0"

def box(ax, x, y, w, h, text, fc, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec="#444", lw=1.0))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                color="#333", lw=1.2))

fig, (axa, axb) = plt.subplots(2, 1, figsize=(10.5, 6.2),
                               gridspec_kw={"height_ratios": [1, 1.5]})

# ---------- Panel (a): simplified ----------
axa.set_xlim(0, 10); axa.set_ylim(0, 2); axa.axis("off")
axa.set_title("(a) Overview — two-pass delta probe", loc="left", fontsize=11, fontweight="bold")
bw, bh = 1.7, 0.9
xs = [0.2, 2.3, 4.4, 6.5, 8.4]
labels = ["full prompt", "frozen LLM", "Δ = pool(full)\n− pool(sys)", "linear probe", "override /\nbenign"]
cols = [C_IN, C_MODEL, C_OPS, C_PROBE, C_DEC]
for x, lab, c in zip(xs, labels, cols):
    box(axa, x, 0.55, bw, bh, lab, c, fs=9)
for i in range(len(xs)-1):
    arrow(axa, xs[i]+bw, 1.0, xs[i+1], 1.0)

# ---------- Panel (b): detailed single-row chain ----------
axb.set_xlim(0, 16); axb.set_ylim(0, 4); axb.axis("off")
axb.set_title("(b) Detailed pipeline (two passes → delta → probe → decision)", loc="left",
              fontsize=11, fontweight="bold")
cy = 1.6  # chain centerline
# stacked inputs
box(axb, 0.1, 2.5, 1.5, 0.8, "system\nprompt", C_IN)
box(axb, 0.1, 0.9, 1.5, 0.8, "full\nprompt", C_IN)
# model
box(axb, 2.2, 1.1, 2.2, 1.8, "Qwen3.5-4B\n(frozen)\n\ntaps {3,6,12}", C_MODEL)
arrow(axb, 1.6, 2.9, 2.2, 2.4)
arrow(axb, 1.6, 1.3, 2.2, 1.6)
# pool
box(axb, 4.9, 1.4, 1.5, 1.2, "pool\nlast-token", C_OPS)
arrow(axb, 4.4, cy, 4.9, cy)
# delta
box(axb, 6.9, 1.4, 1.9, 1.2, "Δ = h_full\n− h_sys\n(concat 7680)", C_OPS, fs=8)
arrow(axb, 6.4, cy, 6.9, cy)
# normalize
box(axb, 9.3, 1.4, 1.5, 1.2, "÷ ‖h_sys‖\nnormalize", C_OPS, fs=8)
arrow(axb, 8.8, cy, 9.3, cy)
# probe
box(axb, 11.3, 1.4, 2.2, 1.2, "Scaler →\nPCA(16) →\nLogReg (bal.)", C_PROBE, fs=8)
arrow(axb, 10.8, cy, 11.3, cy)
# decision
box(axb, 14.0, 1.4, 1.8, 1.2, "score ≥ τ\n(0.258)\n→ OVERRIDE", C_DEC, fs=8, bold=True)
arrow(axb, 13.5, cy, 14.0, cy)

fig.tight_layout()
fig.savefig("fig_method.png", dpi=300, bbox_inches="tight")
print("saved -> fig_method.png")
