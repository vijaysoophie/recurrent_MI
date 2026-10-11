"""Redraw Figs 3 and 4 from the plot data the notebook dumps, to tune the row layout.

The notebook writes figures/figdata_fig3.npz and figures/figdata_fig4.npz beside the
figures themselves. This script redraws both from those dumps so the layout can be
retuned in seconds instead of rerunning the whole notebook. Any change settled here must
be copied back into nb_parts_c.CODE_FIG4 and nb_parts_f.CODE_FIG5, which are what actually
produce the shipped figures.

    python3 preview_rows.py            # writes /tmp/preview_fig3.pdf, /tmp/preview_fig4.pdf
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "figures")

C_MAIN, C_TARGET, C_CTRL = "#2166ac", "#b2182b", "#8c8c8c"
C_INK, C_FAINT, SEQ_CMAP = "#333333", "#c3ced9", "Blues"
WIDE_W, WIDE_H = 7.5, 2.30

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8.0, "axes.labelsize": 8.5, "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
    "mathtext.fontset": "stixsans",
    "lines.linewidth": 1.5, "lines.markersize": 3.5, "lines.markeredgewidth": 0.0,
    "patch.linewidth": 0.6,
    "axes.linewidth": 0.7, "axes.edgecolor": "#8a8a8a", "axes.labelcolor": "#222222",
    "axes.titlepad": 6.0, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "xtick.color": "#4a4a4a", "ytick.color": "#4a4a4a",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5, "legend.labelspacing": 0.35,
    "pdf.fonttype": 42, "figure.dpi": 110, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def seq_cmap(lo=0.25, hi=1.0):
    base = plt.get_cmap(SEQ_CMAP)
    return mpl.colors.LinearSegmentedColormap.from_list(
        "seq_trunc", base(np.linspace(lo, hi, 256)))


def panel(ax, label, dx=-26, dy=4):
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=9.5, fontweight="bold", va="bottom", ha="left",
                color="#111111", annotation_clip=False)


def fig3(W=WIDE_W, H=WIDE_H, out="/tmp/preview_fig3.pdf"):
    d = np.load(os.path.join(FIGDIR, "figdata_fig3.npz"), allow_pickle=True)
    it_hist, mi_hist, pur_hist = d["it_hist"], d["mi_hist"], d["pur_hist"]
    runs = list(zip(d["runs_it"], d["runs_mi"], d["runs_pur"]))
    N_ITER, n_blocks, N_m = int(d["N_ITER"]), int(d["n_blocks"]), int(d["N_m"])
    vmax, W_GRAD = float(d["vmax"]), float(d["W_GRAD"])

    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(1, 5, wspace=0.40, width_ratios=[1, 1, 1, 1, 0.05])
    axA, axB, axC, axD = (fig.add_subplot(gs[0, i]) for i in range(4))
    cax = fig.add_subplot(gs[0, 4])
    for axx in (axA, axB):
        axx.set_box_aspect(1.0)      # same square box as the two matrix panels

    for ith, mih, purh in runs:
        axA.plot(ith + 1, mih, color=C_FAINT, lw=0.7, zorder=1)
    axA.plot(it_hist + 1, mi_hist, color=C_MAIN, lw=1.5, zorder=3, label="best restart")
    axA.set_xscale("log")
    axA.axhline(float(d["mi_match"]), color=C_TARGET, ls="--", lw=1.0, label="matched")
    axA.axhline(float(d["mi_rand"]), color=C_CTRL, ls="-.", lw=1.0, label="random")
    axA.axhline(float(d["I_ff"]), color=C_INK, ls=":", lw=1.0, label="no inhibition")
    axA.set_xlabel("iteration + 1"); axA.set_ylabel("$I(\\mathbf{c},\\mathbf{y})$  (nats)")
    axA.set_title(f"ascent at $w={W_GRAD}$", fontsize=8.0)
    axA.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=5.8,
                 labelspacing=0.25)

    for ith, mih, purh in runs:
        axB.plot(ith + 1, purh, color=C_FAINT, lw=0.7, zorder=1)
    axB.plot(it_hist + 1, pur_hist, color=C_MAIN, lw=1.5, zorder=3)
    axB.set_xscale("log")
    axB.axhline(1.0, color=C_TARGET, ls="--", lw=1.0)
    axB.axhline(float(d["chance"]), color=C_CTRL, ls="-.", lw=1.0)
    axB.text(N_ITER, 1.0, "matched", ha="right", va="bottom", fontsize=6.2, color=C_TARGET)
    axB.text(N_ITER, float(d["chance"]), "chance", ha="right", va="bottom",
             fontsize=6.2, color=C_CTRL)
    axB.set_ylim(0, 1.12)
    axB.set_xlabel("iteration + 1"); axB.set_ylabel("block purity")
    axB.set_title("granule cells concentrate", fontsize=8.0)

    for axx, K, ttl in ((axC, d["K_before"], "before"), (axD, d["K_after"], "after")):
        im = axx.imshow(K, cmap=seq_cmap(0.0, 1.0), interpolation="nearest",
                        vmin=0.0, vmax=vmax)
        for b in range(1, n_blocks):
            axx.axhline(b * N_m / n_blocks - 0.5, color="#d33", lw=0.6)
            axx.axvline(b * N_m / n_blocks - 0.5, color="#d33", lw=0.6)
        axx.set_title(f"$\\mathbf{{W}}^{{(gm)\\top}}\\mathbf{{W}}^{{(gm)}}$, {ttl}",
                      fontsize=8.0)
        axx.set_xlabel("mitral cell"); axx.grid(False)
        axx.tick_params(labelsize=6.5)
    axC.set_ylabel("mitral cell")
    axD.set_yticklabels([])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6.0)

    for axx, lab in ((axA, "a"), (axB, "b"), (axC, "c"), (axD, "d")):
        panel(axx, lab)
    fig.subplots_adjust(left=0.055, right=0.955, bottom=0.20, top=0.86)
    # the square panels are shorter than their gridspec slot, so match the colorbar to them
    fig.canvas.draw()
    pd, pc = axD.get_position(), cax.get_position()
    cax.set_position([pc.x0, pd.y0, pc.width, pd.height])
    fig.savefig(out)
    print("wrote", out)


def fig4(W=WIDE_W, H=WIDE_H, out="/tmp/preview_fig4.pdf"):
    d = np.load(os.path.join(FIGDIR, "figdata_fig4.npz"), allow_pickle=True)
    WS, CH_W, Iff = d["CH_WS"], float(d["CH_W"]), float(d["ch_Iff"])

    fig, axes = plt.subplots(1, 4, figsize=(W, H))
    ax = axes.ravel()

    labs = ["input", "pub.", "grown", "random"]
    vals = list(d["top"])
    ax[0].bar(range(4), vals, color=[C_FAINT, C_FAINT, C_TARGET, C_CTRL], width=0.62)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.4)
    ax[0].set_xticks(range(4), labs, fontsize=6.8)
    ax[0].set_ylim(0, 1.08); ax[0].grid(axis="x", visible=False)
    ax[0].set_ylabel("enantiomer-pair corr.")
    ax[0].set_title(f"their decorrelation, $w={CH_W}$", fontsize=8.0)

    ax[1].plot(WS[1:], d["IA_a"][1:], color=C_TARGET, label="grown")
    ax[1].plot(WS[1:], d["IA_r"][1:], color=C_CTRL, ls="--", label="random")
    ax[1].axhline(d["IA_a"][0], color=C_INK, ls=":", lw=1.0, label="no inhib.")
    ax[1].axvline(CH_W, color="#9a9a9a", lw=0.8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("inhibitory strength $w$")
    ax[1].set_ylabel("$I(\\mathbf{c},\\mathbf{M})$  (nats)")
    ax[1].set_title("Model A: in-loop noise", fontsize=8.0)
    ax[1].legend(loc="lower left", fontsize=6.0, labelspacing=0.25)

    ax[2].plot(WS[1:], d["IB_a"][1:] - Iff, color=C_TARGET, label="grown")
    ax[2].plot(WS[1:], d["IB_r"][1:] - Iff, color=C_CTRL, ls="--", label="random")
    ax[2].plot([CH_W], [float(d["ch_gain_grad"])], "o", color=C_MAIN, ms=4.5, zorder=5,
               label="grad. ascent")
    ax[2].axhline(float(d["ch_avail"]), color=C_MAIN, ls="-.", lw=1.0, label="whitened")
    ax[2].axhline(0.0, color=C_INK, ls=":", lw=1.0)
    ax[2].axvline(CH_W, color="#9a9a9a", lw=0.8)
    ax[2].set_xscale("log"); ax[2].set_xlabel("inhibitory strength $w$")
    ax[2].set_ylabel("$I(w)-I(0)$  (nats)")
    ax[2].set_title("Model B: readout noise, budget", fontsize=8.0)
    ax[2].legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=6.0, labelspacing=0.25)

    ax[3].plot(d["gmins"], d["gfrac"], "o-", color=C_TARGET, ms=3.5)
    ax[3].axvline(1.2, color="#9a9a9a", lw=0.8)
    ax[3].text(1.23, 6, "Chow et al.", fontsize=6.2, color="#666666")
    ax[3].set_ylim(0, 100)
    ax[3].set_xlabel("survival threshold $G_{\\min}$")
    ax[3].set_ylabel("Model B gain (% of whitened)")
    ax[3].set_title("selectivity of neurogenesis", fontsize=8.0)

    for axx in ax:
        axx.tick_params(labelsize=6.8)
    for axx, lab in zip(ax, "abcd"):
        panel(axx, lab)
    fig.tight_layout(w_pad=0.9)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    fig3()
    fig4()
