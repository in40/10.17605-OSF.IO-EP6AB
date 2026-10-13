"""
make_method_figure_ru.py — RUSSIAN-labelled two-panel method scheme.
Run locally (no data dependency). Uses DejaVu Sans for Cyrillic.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
C_IN = "#e8f0fe"; C_MODEL = "#dbe9d5"; C_OPS = "#fde8e8"; C_PROBE = "#fff3d6"; C_DEC = "#e5e0f0"

def box(ax, x, y, w, h, text, fc, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec="#444", lw=1.0))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color="#333", lw=1.2))

fig, (axa, axb) = plt.subplots(2, 1, figsize=(10.5, 6.2), gridspec_kw={"height_ratios": [1, 1.5]})

# Panel (a)
axa.set_xlim(0, 10); axa.set_ylim(0, 2); axa.axis("off")
axa.set_title("(а) Общий вид: двухпроходный дельта-зонд", loc="left", fontsize=11, fontweight="bold")
bw, bh = 1.7, 0.9
xs = [0.2, 2.3, 4.4, 6.5, 8.4]
labels = ["полный промпт", "замороженная\nLLM", "Δ = pool(полн.)\n− pool(сист.)", "линейный\nзонд", "переопр. /\nдоброкач."]
cols = [C_IN, C_MODEL, C_OPS, C_PROBE, C_DEC]
for x, lab, c in zip(xs, labels, cols):
    box(axa, x, 0.55, bw, bh, lab, c, fs=8.5)
for i in range(len(xs)-1):
    arrow(axa, xs[i]+bw, 1.0, xs[i+1], 1.0)

# Panel (b)
axb.set_xlim(0, 16); axb.set_ylim(0, 4); axb.axis("off")
axb.set_title("(б) Подробный конвейер обработки (два прохода → дельта → зонд → решение)", loc="left", fontsize=11, fontweight="bold")
cy = 1.6
box(axb, 0.1, 2.5, 1.5, 0.8, "системный\nпромпт", C_IN)
box(axb, 0.1, 0.9, 1.5, 0.8, "полный\nпромпт", C_IN)
box(axb, 2.2, 1.1, 2.2, 1.8, "Qwen3.5-4B\n(заморожена)\n\nотводы {3,6,12}", C_MODEL)
arrow(axb, 1.6, 2.9, 2.2, 2.4)
arrow(axb, 1.6, 1.3, 2.2, 1.6)
box(axb, 4.9, 1.4, 1.5, 1.2, "агрегация\nпосл. токена", C_OPS)
arrow(axb, 4.4, cy, 4.9, cy)
box(axb, 6.9, 1.4, 1.9, 1.2, "Δ = h_полн\n− h_сист\n(7680)", C_OPS, fs=8)
arrow(axb, 6.4, cy, 6.9, cy)
box(axb, 9.3, 1.4, 1.5, 1.2, "÷ ‖h_сист‖\nнормировка", C_OPS, fs=8)
arrow(axb, 8.8, cy, 9.3, cy)
box(axb, 11.3, 1.4, 2.2, 1.2, "Scaler →\nPCA(16) →\nLogReg (бал.)", C_PROBE, fs=8)
arrow(axb, 10.8, cy, 11.3, cy)
box(axb, 14.0, 1.4, 1.8, 1.2, "балл ≥ τ\n(0.258)\n→ ПЕРЕОПРЕД.", C_DEC, fs=8, bold=True)
arrow(axb, 13.5, cy, 14.0, cy)

fig.tight_layout()
fig.savefig("fig_method_ru.png", dpi=300, bbox_inches="tight")
print("saved -> fig_method_ru.png")
