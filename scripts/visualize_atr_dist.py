"""
ATR Multiple Distribution Visualizer
=====================================
x-axis : LONG  방향 excursion (bar의 high - prev_close) / ATR
y-axis : SHORT 방향 excursion (prev_close - bar's low)  / ATR
z-axis : frequency (%)

슬라이더로 TP/SL n 값을 조정하면:
  - 선택된 TP/SL에서 LONG/SHORT Win Probability 실시간 업데이트
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")          # interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Patch

# ── 1. Load & compute ─────────────────────────────────────────────
DATA_PATH = "data/training_BTCUSDT_15m_20220101.csv"
print("[load] Reading data...")
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df.columns = [c.lower() for c in df.columns]

high       = df["high"].values
low        = df["low"].values
close      = df["close"].values
prev_close = np.roll(close, 1);  prev_close[0] = close[0]

tr  = np.maximum(high - low,
      np.maximum(np.abs(high - prev_close),
                 np.abs(low  - prev_close)))
atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values

# Per-bar excursion from previous close (in ATR units)
n_up   = np.clip((high[1:]  - prev_close[1:]) / atr[1:], 0, 5.0)
n_down = np.clip((prev_close[1:] - low[1:])   / atr[1:], 0, 5.0)
N      = len(n_up)
print(f"[data] {N:,} bars  |  n_up P50={np.percentile(n_up,50):.3f}  n_down P50={np.percentile(n_down,50):.3f}")

# ── 2. 2-D histogram (bins 0…5 step 0.2) ─────────────────────────
STEP   = 0.2
BINS   = np.arange(0, 5.0 + STEP, STEP)
H, xe, ye = np.histogram2d(n_up, n_down, bins=BINS)
H_pct = H / N * 100          # % of all bars

xc = (xe[:-1] + xe[1:]) / 2  # bin centres
yc = (ye[:-1] + ye[1:]) / 2

# ── 3. Build figure ───────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor="#0d0d0d")
fig.suptitle("15m BTCUSDT  |  Per-bar ATR Excursion Distribution  (2022-01-01 ~ 2025-10-01)",
             color="white", fontsize=13, y=0.97)

# 3-D axes
ax3d = fig.add_axes([0.03, 0.18, 0.58, 0.75], projection="3d")
ax3d.set_facecolor("#0d0d0d")
ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False
for spine in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
    spine.pane.set_edgecolor("#333")

# Stats panel (right)
ax_stat = fig.add_axes([0.63, 0.30, 0.34, 0.60])
ax_stat.set_facecolor("#111")
ax_stat.axis("off")

# Marginal histogram axes (right lower)
ax_marg = fig.add_axes([0.63, 0.18, 0.34, 0.12])
ax_marg.set_facecolor("#111")

# ── 4. Draw 3-D bars ──────────────────────────────────────────────
XM, YM = np.meshgrid(xc, yc, indexing="ij")
xpos   = XM.flatten()
ypos   = YM.flatten()
zpos   = np.zeros_like(xpos)
dz     = H_pct.flatten()

norm   = plt.Normalize(vmin=0, vmax=dz.max())
colors = cm.plasma(norm(dz))

ax3d.bar3d(xpos - STEP/2, ypos - STEP/2, zpos,
           STEP * 0.88, STEP * 0.88, dz,
           color=colors, alpha=0.85, shade=True)

ax3d.set_xlabel("LONG excursion  (n×ATR up)", color="cyan",   labelpad=8)
ax3d.set_ylabel("SHORT excursion (n×ATR down)", color="orange", labelpad=8)
ax3d.set_zlabel("Frequency (%)", color="white", labelpad=6)
ax3d.tick_params(colors="white")
ax3d.set_xlim(0, 5); ax3d.set_ylim(0, 5)

# TP / SL lines on the floor
tp_init, sl_init = 4.0, 1.0
long_tp_line,  = ax3d.plot([tp_init, tp_init], [0, 5],    [0, 0], "c--",  lw=2, label="TP (LONG )")
long_sl_line,  = ax3d.plot([0, 5],    [sl_init, sl_init], [0, 0], "r--",  lw=2, label="SL (vs LONG)")
ax3d.legend(loc="upper right", fontsize=8, facecolor="#222", edgecolor="#555",
            labelcolor="white")

# ── 5. Marginal bar chart (|n| distribution) ─────────────────────
bins1d = BINS
long_hist,  _ = np.histogram(n_up,   bins=bins1d)
short_hist, _ = np.histogram(n_down, bins=bins1d)
bx = (bins1d[:-1] + bins1d[1:]) / 2

ax_marg.bar(bx - 0.05, long_hist  / N * 100, width=0.18,
            color="cyan",   alpha=0.7, label="LONG(up)")
ax_marg.bar(bx + 0.05, short_hist / N * 100, width=0.18,
            color="orange", alpha=0.7, label="SHORT(down)")
ax_marg.set_facecolor("#111")
ax_marg.tick_params(colors="white", labelsize=8)
ax_marg.set_xlabel("n×ATR", color="white", fontsize=8)
ax_marg.set_ylabel("%", color="white", fontsize=8)
ax_marg.legend(fontsize=7, facecolor="#222", labelcolor="white", edgecolor="#444")
ax_marg.spines[:].set_color("#333")
# mark current TP / SL
tp_vline = ax_marg.axvline(tp_init, color="cyan",   ls="--", lw=1.5, label="TP")
sl_vline = ax_marg.axvline(sl_init, color="red",    ls="--", lw=1.5, label="SL")

# ── 6. Stats text helper ──────────────────────────────────────────
def win_prob(tp, sl):
    """
    LONG:  win = n_up>=tp AND n_down<sl  (TP hit before SL)
    SHORT: win = n_down>=tp AND n_up<sl
    """
    long_win  = np.mean((n_up >= tp) & (n_down < sl)) * 100
    short_win = np.mean((n_down >= tp) & (n_up < sl)) * 100
    both_hit  = np.mean((n_up >= tp) & (n_down >= sl)) * 100
    no_hit    = np.mean((n_up < tp)  & (n_down < sl))  * 100
    return long_win, short_win, both_hit, no_hit

def bep(tp, sl, fee_pct=0.00075, leverage=10, pos_frac=0.5):
    eff = leverage * pos_frac
    gross_win  = tp * (atr[len(atr)//2] / close[len(close)//2]) * eff   # approx
    gross_loss = sl * (atr[len(atr)//2] / close[len(close)//2]) * eff
    net_win    = gross_win  - fee_pct * eff
    net_loss   = gross_loss + fee_pct * eff
    if (net_win + net_loss) == 0:
        return 50.0
    return net_loss / (net_win + net_loss) * 100

stat_text = None

def update_stats(tp, sl):
    global stat_text
    lw, sw, bh, nh = win_prob(tp, sl)
    bp = sl / (tp + sl) * 100          # theoretical gambler's ruin BEP
    rr = tp / sl if sl > 0 else float("inf")

    # Fee-adjusted BEP (approximate with median ATR/price ratio)
    mid = len(close) // 2
    atr_frac = float(np.median(atr[mid-1000:mid] / close[mid-1000:mid]))
    eff = 10 * 0.5
    net_w = tp * atr_frac * eff - 0.00075 * eff
    net_l = sl * atr_frac * eff + 0.00075 * eff
    fee_bep = net_l / (net_w + net_l) * 100 if (net_w + net_l) > 0 else 50

    lines = [
        "  TP / SL Settings",
        f"  TP = {tp:.2f}×ATR",
        f"  SL = {sl:.2f}×ATR",
        f"  R:R ratio = {rr:.1f}:1",
        "",
        "  Theoretical (gambler's ruin)",
        f"  BEP (gross)      = {bp:.1f}%",
        f"  BEP (w/ fees)    = {fee_bep:.1f}%",
        "",
        "  Per-bar Win Probability",
        f"  LONG  wins (this bar) = {lw:.1f}%",
        f"  SHORT wins (this bar) = {sw:.1f}%",
        f"  Both hit same bar    = {bh:.1f}%",
        f"  Neither hit          = {nh:.1f}%",
        "",
        "  Single-bar excursion stats",
        f"  P(n_up   >= {tp:.1f}) = {np.mean(n_up>=tp)*100:.1f}%",
        f"  P(n_down >= {tp:.1f}) = {np.mean(n_down>=tp)*100:.1f}%",
        f"  P(n_up   >= {sl:.1f}) = {np.mean(n_up>=sl)*100:.1f}%",
        f"  P(n_down >= {sl:.1f}) = {np.mean(n_down>=sl)*100:.1f}%",
        "",
        "  Practical Note",
        f"  Holding ~{int(1/max(lw/100,1e-4))} bars avg to get LONG TP",
        f"  Median bar excursion = {np.median(n_up):.3f}×ATR",
    ]

    if stat_text is not None:
        stat_text.remove()
    stat_text = ax_stat.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax_stat.transAxes,
        fontsize=9, color="white",
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(facecolor="#1a1a2e", edgecolor="#444", boxstyle="round,pad=0.5")
    )
    fig.canvas.draw_idle()

# ── 7. Sliders ─────────────────────────────────────────────────────
ax_tp = fig.add_axes([0.07, 0.10, 0.45, 0.025], facecolor="#1a1a2e")
ax_sl = fig.add_axes([0.07, 0.05, 0.45, 0.025], facecolor="#1a1a2e")

s_tp = Slider(ax_tp, "TP (n×ATR)", 0.5, 5.0, valinit=tp_init, valstep=0.1, color="cyan")
s_sl = Slider(ax_sl, "SL (n×ATR)", 0.1, 3.0, valinit=sl_init, valstep=0.1, color="red")

for s in [s_tp, s_sl]:
    s.label.set_color("white")
    s.valtext.set_color("white")

def on_change(val):
    tp = s_tp.val
    sl = s_sl.val
    long_tp_line.set_xdata([tp, tp])
    long_sl_line.set_ydata([sl, sl])
    tp_vline.set_xdata([tp, tp])
    sl_vline.set_xdata([sl, sl])
    update_stats(tp, sl)

s_tp.on_changed(on_change)
s_sl.on_changed(on_change)

# ── 8. Reset button ───────────────────────────────────────────────
ax_btn = fig.add_axes([0.55, 0.055, 0.07, 0.04])
btn = Button(ax_btn, "Reset", color="#222", hovercolor="#444")
btn.label.set_color("white")

def reset(event):
    s_tp.reset(); s_sl.reset()
btn.on_clicked(reset)

# ── 9. Color bar ──────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax3d, shrink=0.4, pad=0.04)
cbar.ax.tick_params(colors="white", labelsize=7)
cbar.set_label("% of bars", color="white", fontsize=8)

# Initial stats
update_stats(tp_init, sl_init)

print("[viz] Showing interactive window. Use sliders to adjust TP/SL.")
print("      Rotate 3D chart by clicking and dragging.")
plt.show()
