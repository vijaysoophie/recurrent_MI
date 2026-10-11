"""Cell sources for recurrent_MI_simulations.ipynb, part A: setup through Fig 1."""

MD_TITLE = r"""
# Simulations for *Mutual information in linear recurrent circuits: inhibition in the olfactory bulb increases information only against a fixed noise floor*

Vijay Singh and Martin Tchernookov.

This notebook reproduces every figure and every number reported in the manuscript and
its supporting information (S1 Text). It runs top to bottom with no arguments, and it prints, beside each
computed value, the value quoted in the paper, so that a mismatch is visible immediately
rather than requiring a manual audit.

| output | file | where it appears |
|---|---|---|
| Fig 1 | `figures/Fig1_framework.pdf` | main text, the framework |
| Fig 2 | `figures/Fig2_two_noise_models.pdf` | main text, the two noise models compared |
| Fig 3 | `figures/Fig4_gradient_ascent.pdf` | main text, matched wiring is found by optimization |
| Fig 4 | `figures/Fig5_chow_network.pdf` | main text, the analysis on Chow et al.'s network |
| Appendix figure | `figures/Fig3_flatness.pdf` | appendix, accuracy of the spectral-flatness identity |
| `results.json` | machine-readable dump of every recorded quantity | used to fill and audit the manuscript |

**Data.** One data file is used: `data/chow2012_stimuli.npz`, the eight glomerular
activation patterns of Chow, Wick & Riecke (2012), recovered from their published figures.
`chow_model.py` is the engine of our reproduction of that paper and simulates their
neurogenesis rule; it is used only in section 9. Everything else is synthetic.
"""

CODE_DEPS = r'''
import os, json, time, platform
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import solve_continuous_lyapunov

print("python     ", platform.python_version())
print("numpy      ", np.__version__)
print("scipy      ", scipy.__version__)
print("matplotlib ", mpl.__version__)
print()
print("All results below are computed in closed form or by an exact linear solve, with the")
print("single exception of the Monte-Carlo validation in section 2, which is stochastic by")
print("design and is seeded. Quantities that depend on an average over random connectivity")
print("seeds are reported with the seed count and the standard error of that average.")
'''

CODE_STYLE = r'''
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

FULL_W = 7.2                       # authoring width for a width=\textwidth figure
# Figs 3 and 4 are single rows of four panels, typeset across the manuscript's left margin
# with the PLOS adjustwidth idiom: text width 5.25in plus 2.25in taken from that margin.
# WIDE_W must match the \includegraphics width used for them in manuscript_template.tex.
WIDE_W, WIDE_H = 7.5, 2.30

def fig_width(frac=1.0):
    return FULL_W * frac

# Colours are assigned by meaning, not by figure, so the same condition wears the same
# colour throughout: red is the matched / theoretical target, grey is the unstructured
# control, blue is the quantity the paper is actually tracking, green is a fourth series.
C_MAIN   = "#2166ac"
C_TARGET = "#b2182b"
C_CTRL   = "#8c8c8c"
C_ALT    = "#1b7837"
C_INK    = "#333333"
C_FAINT  = "#c3ced9"
SEQ_CMAP = "Blues"

def seq_cmap(lo=0.25, hi=1.0):
    base = plt.get_cmap(SEQ_CMAP)
    return mpl.colors.LinearSegmentedColormap.from_list(
        "seq_trunc", base(np.linspace(lo, hi, 256)))

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8.0, "axes.labelsize": 8.5, "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
    "mathtext.fontset": "stixsans",
    "lines.linewidth": 1.5, "lines.markersize": 3.5, "lines.markeredgewidth": 0.0,
    "patch.linewidth": 0.6,
    "axes.linewidth": 0.7, "axes.edgecolor": "#8a8a8a", "axes.labelcolor": "#222222",
    "axes.titlepad": 6.0, "axes.titleweight": "normal",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "grid.alpha": 1.0,
    "xtick.color": "#4a4a4a", "ytick.color": "#4a4a4a",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5, "legend.labelspacing": 0.35,
    # embed TrueType rather than Type 3 fonts, which some journals reject
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 110, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

def panel(ax, label, dx=-30, dy=8):
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=9.5, fontweight="bold", va="bottom", ha="left",
                color="#111111", annotation_clip=False)

def panels(axes, labels="abcdefgh", **kw):
    for ax, lab in zip(np.ravel(np.asarray(axes, dtype=object)), labels):
        panel(ax, lab, **kw)

def save(fig, name):
    """Write a vector PDF (the file the manuscript includes) and a PNG preview."""
    pdf_path = os.path.join(OUTDIR, os.path.splitext(name)[0] + ".pdf")
    png_path = os.path.join(OUTDIR, os.path.splitext(name)[0] + ".png")
    fig.savefig(pdf_path); fig.savefig(png_path)
    w, h = fig.get_size_inches()
    print(f"  -> {pdf_path}   ({w:.2f} x {h:.2f} in)")
    return pdf_path

# ---------------------------------------------------------------- audit machinery
RESULTS = {}
CHECKS = []          # (label, computed, quoted, passed) for every check() call
FAILURES = []

def record(key, value):
    """Store a quantity for the machine-readable audit dump at the end."""
    RESULTS[key] = float(value)
    return value

def check(label, got, want, tol=0.02, fmt="{:.3f}", key=None):
    """Compare a computed value against the value printed in the manuscript.

    Every call is logged. The manifest cell at the end of the notebook raises if any
    comparison failed, so a mismatch stops the run rather than scrolling past."""
    got = float(got)
    if key is None:
        key = label
    record(key, got)
    if want is None or (isinstance(want, float) and np.isnan(want)):
        CHECKS.append((label, got, None, None))
        print(f"  [--- ] {label:<50s} computed {fmt.format(got):>11s}   (no reference yet)")
        return got
    passed = abs(got - want) <= tol * max(1.0, abs(want))
    CHECKS.append((label, got, want, passed))
    if not passed:
        FAILURES.append(f"{label}: computed {got!r}, manuscript {want!r}")
    print(f"  [{'OK  ' if passed else 'DIFF'}] {label:<50s} computed {fmt.format(got):>11s}   "
          f"manuscript {fmt.format(want):>11s}")
    return got

def show_table(rows, header, title=None):
    widths = [max(len(str(header[i])), *(len(str(r[i])) for r in rows)) + 2
              for i in range(len(header))]
    line = "".join(str(h).ljust(w) for h, w in zip(header, widths)).rstrip()
    if title:
        print(title); print("=" * len(line))
    print(line); print("-" * len(line))
    for r in rows:
        print("".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip())

# a relative path, so the saved output does not carry whatever absolute path the
# notebook happened to be run from
print("style applied; figures ->", os.path.relpath(OUTDIR))
'''

MD_SETUP = r"""
## 1. The world, the two mutual-information formulas, and the odor ensemble

Everything below is built on one network and one stimulus ensemble, defined once here.

**The network.** $N_m = 50$ mitral cells and $N_g = 200$ granule cells coupled through
reciprocal dendrodendritic synapses, so that the effective mitral-to-mitral coupling is
$\mathbf W_{\mathrm{eff}} = \mathbf I + w\,\mathbf W^{(gm)\top}\mathbf W^{(gm)}$. A granule
cell is a binary row of $\mathbf W^{(gm)}$ carrying $n_{\mathrm{connect}}$ unit entries. In
*random* wiring those targets are drawn uniformly from all mitral cells; in
*source-matched* wiring they are drawn from a single odor-source block.

**The stimulus.** A block-structured covariance $\Sigma_c$ standing for five odor sources:
each source occupies a block of ten glomeruli with within-block correlation $0.6$ and unit
variance, on top of a background variance $0.2$ shared by every cell. Two glomeruli are
correlated if and only if they are driven by the same source.

**The two noise conventions.** Model A places the noise inside the recurrent loop, so the
recurrence filters it exactly as it filters the signal, and the Lyapunov solution for
symmetric $\mathbf W_{\mathrm{eff}}$ is $\Sigma_{r\mid c} = \tfrac{\sigma^2}{2}\mathbf
W_{\mathrm{eff}}^{-1}$. Model B places the noise downstream at a readout and holds total
output power at the feedforward level $P = \operatorname{Tr}\Sigma_c$ with a compensating
gain $g$. The two give opposite verdicts on recurrent inhibition, which is the dichotomy
the paper is organized around.

**Spectral flatness.** $\mathcal F(\Sigma) = (\det\Sigma)^{1/N} / (\operatorname{Tr}\Sigma/N)$,
the ratio of the geometric to the arithmetic mean of the eigenvalues, equal to one exactly
when the spectrum is white. It is defined here and used in section 5.
"""

CODE_SETUP = r'''
# ---------------------------------------------------------------- world
N_m, n_blocks = 50, 5
block_corr, within_var, background_var = 0.6, 1.0, 0.2
sigma_noise = 0.1                       # readout noise sigma_out, and the in-loop sigma
blocks = np.repeat(np.arange(n_blocks), N_m // n_blocks)
SEED = 1234

def generate_odor_covariance(N=N_m, nb=n_blocks, corr=block_corr,
                             var=within_var, bg=background_var):
    """Block covariance: each block is one correlated odor source."""
    bs = N // nb
    Sig = bg * np.eye(N)
    for b in range(nb):
        lo, hi = b * bs, min((b + 1) * bs, N)
        blk = np.full((hi - lo, hi - lo), var * corr)
        np.fill_diagonal(blk, var)
        Sig[lo:hi, lo:hi] += blk
    return Sig

Sigma_c = generate_odor_covariance()
power_budget = np.trace(Sigma_c)

within_pair = Sigma_c[0, 1]
across_pair = Sigma_c[0, N_m // n_blocks]
assert np.isclose(within_pair, block_corr) and np.isclose(across_pair, 0.0)
print(f"block covariance: within-block {within_pair:.2f}, cross-block {across_pair:.2f}, "
      f"diagonal {Sigma_c[0,0]:.2f}")

# ---------------------------------------------------------------- the two MI formulas
def mi_inloop(Sigma_c, W_eff, sigma):
    """Model A: noise inside the recurrent loop."""
    n = W_eff.shape[0]
    return 0.5 * np.linalg.slogdet(
        np.eye(n) + (2.0 / sigma**2) * np.linalg.solve(W_eff, Sigma_c))[1]

def output_covariance(Sigma_c, W_eff):
    """C_w = W_eff^{-1} Sigma_c W_eff^{-T}, the output signal covariance before the gain."""
    F = np.linalg.inv(W_eff)
    return F @ Sigma_c @ F.T

def mi_budget(Sigma_c, W_eff, sigma, budget=None):
    """Model B: readout noise downstream of the filter, plus a fixed activity budget."""
    C = output_covariance(Sigma_c, W_eff)
    P = np.trace(Sigma_c) if budget is None else budget
    g2 = P / np.trace(C)
    return 0.5 * np.linalg.slogdet(np.eye(W_eff.shape[0]) + g2 * C / sigma**2)[1]

def W_eff_from(W_gm, w):
    return np.eye(W_gm.shape[1]) + w * (W_gm.T @ W_gm)

def spectral_flatness(Sig):
    """F(Sigma) = geometric mean / arithmetic mean of the eigenvalues; 1 iff white."""
    ev = np.linalg.eigvalsh(Sig)
    ev = np.maximum(ev, 1e-300)
    return float(np.exp(np.mean(np.log(ev))) / np.mean(ev))

def block_purity(W_gm, blk=blocks):
    """For each granule cell, the largest fraction of its targets falling in one block,
    averaged over cells with at least one target."""
    n_b = int(blk.max()) + 1
    pur = [max(W_gm[j][blk == b].sum() for b in range(n_b)) / W_gm[j].sum()
           for j in range(W_gm.shape[0]) if W_gm[j].sum() > 0]
    return float(np.mean(pur)) if pur else np.nan

# ---------------------------------------------------------------- wiring
def make_wiring(N_m_, N_g_, n_connect, kind, blk=None, seed=0):
    """Binary granule-to-mitral connectivity. 'random' draws targets uniformly;
    'matched' draws every target of a given granule cell from one odor-source block."""
    rng = np.random.default_rng(seed)
    W = np.zeros((N_g_, N_m_))
    if kind == "matched":
        n_b = int(blk.max()) + 1
        for j in range(N_g_):
            pool = np.where(blk == rng.integers(n_b))[0]
            W[j, rng.choice(pool, min(n_connect, len(pool)), replace=False)] = 1.0
    else:
        for j in range(N_g_):
            W[j, rng.choice(N_m_, n_connect, replace=False)] = 1.0
    return W

# ---------------------------------------------------------------- odor ensemble
N_ODORS = 200                # up from the 40 used in earlier drafts
rng_ens = np.random.default_rng(7)
sources = np.zeros((n_blocks, N_m))
for b in range(n_blocks):
    pool = np.where(blocks == b)[0]
    sources[b, pool] = 1.0 + 0.4 * rng_ens.random(len(pool))

odors = np.zeros((N_m, N_ODORS))
odor_source = np.zeros(N_ODORS, dtype=int)
for k in range(N_ODORS):
    b = int(rng_ens.integers(n_blocks))
    odor_source[k] = b
    mix = 0.15 * rng_ens.random(n_blocks)
    mix[b] = 1.0
    odors[:, k] = np.maximum(sources.T @ mix + 0.1 * rng_ens.standard_normal(N_m), 0)

pairs = np.triu_indices(N_ODORS, k=1)
same_src = (odor_source[:, None] == odor_source[None, :])[pairs]
r_in_same = float(np.corrcoef(odors.T)[pairs][same_src].mean())

I_ff = mi_budget(Sigma_c, np.eye(N_m), sigma_noise, power_budget)
F_in = spectral_flatness(Sigma_c)

print(f"power budget P = tr(Sigma_c) = {power_budget:.1f}")
print(f"{N_ODORS} odors drawn from {n_blocks} sources")
print()
check("same-source input correlation", r_in_same, REF_r_in_same, fmt="{:.3f}",
      key="r_in_same")
check("feedforward MI I(0), nats", I_ff, REF_I_ff, fmt="{:.3f}", key="I_ff")
check("input spectral flatness F(Sigma_c)", F_in, REF_F_in, fmt="{:.4f}", key="F_in")
pearson_within = float(Sigma_c[0, 1] / Sigma_c[0, 0])
print(f"within-source Pearson correlation of Sigma_c = {pearson_within:.2f} "
      f"(covariance {Sigma_c[0,1]:.2f} over variance {Sigma_c[0,0]:.2f})")
check("within-source Pearson correlation", pearson_within, REF_pearson, fmt="{:.3f}",
      key="pearson_within")
record("sigma_c_diag", Sigma_c[0, 0])
for k_, v_ in (("P_budget", power_budget), ("N_ODORS", N_ODORS), ("N_m", N_m),
               ("n_blocks", n_blocks), ("sigma_out", sigma_noise),
               ("block_size", N_m // n_blocks)):
    record(k_, v_)
'''

MD_MC = r"""
## 2. Monte-Carlo validation of the closed-form mutual information

Every number in this paper comes from a determinant rather than from a simulation, so the
determinant had better be right. This section checks it directly. We integrate the
stochastic dynamics

$$d\mathbf r = (\mathbf W\mathbf r + \mathbf S\mathbf c)\,dt + \Sigma_\eta^{1/2}\,d\mathbf B_t$$

for a small network with the Euler-Maruyama scheme, collect the stationary response over
many trials and many stimuli, and compare three things against theory: the conditional
covariance against the Lyapunov solution, the marginal covariance against the law of total
covariance, and the mutual information estimated from the two sampled covariances against
the closed form. The test is run on a small network, $N = 8$, so that the sampled
covariances converge at a tractable number of trials.

This is the only stochastic computation in the notebook.
"""

CODE_MC = r'''
def simulate_ou(W, S, Sigma_eta, c, dt, n_steps, n_trials, burn, rng):
    """Euler-Maruyama integration of dr = (W r + S c) dt + Sigma_eta^{1/2} dB,
    returning the stationary samples after a burn-in. One fixed stimulus c."""
    N = W.shape[0]
    L = np.linalg.cholesky(Sigma_eta)
    r = np.zeros((n_trials, N))
    drive = (S @ c)[None, :]
    out = np.empty((n_trials, n_steps - burn, N))
    sq = np.sqrt(dt)
    for t in range(n_steps):
        noise = rng.standard_normal((n_trials, N)) @ L.T
        r = r + dt * (r @ W.T + drive) + sq * noise
        if t >= burn:
            out[:, t - burn, :] = r
    return out

# ---- a small, generic, non-symmetric test network
rng_mc = np.random.default_rng(SEED)
N_mc, M_mc = 8, 4
W_mc = -np.eye(N_mc) + 0.25 * rng_mc.standard_normal((N_mc, N_mc))
W_mc -= np.eye(N_mc) * max(0.0, 0.05 + np.max(np.linalg.eigvals(W_mc).real))
S_mc = rng_mc.standard_normal((N_mc, M_mc)) / np.sqrt(M_mc)
A_mc = rng_mc.standard_normal((M_mc, M_mc))
Sigma_c_mc = A_mc @ A_mc.T / M_mc + 0.2 * np.eye(M_mc)
B_mc = rng_mc.standard_normal((N_mc, N_mc))
Sigma_eta_mc = 0.3 * (B_mc @ B_mc.T / N_mc + np.eye(N_mc))

assert np.max(np.linalg.eigvals(W_mc).real) < 0, "test network must be stable"

# ---- theory
Src_theory = solve_continuous_lyapunov(W_mc, -Sigma_eta_mc)
Winv_S = np.linalg.solve(W_mc, S_mc)
Sr_theory = Src_theory + Winv_S @ Sigma_c_mc @ Winv_S.T
I_theory = 0.5 * (np.linalg.slogdet(Sr_theory)[1] - np.linalg.slogdet(Src_theory)[1])

# ---- Monte Carlo
dt_mc, n_steps_mc, burn_mc, n_trials_mc, n_stim_mc = 0.004, 3000, 1500, 300, 1200
t0 = time.time()
cs = rng_mc.multivariate_normal(np.zeros(M_mc), Sigma_c_mc, size=n_stim_mc)
cond_cov = np.zeros((N_mc, N_mc))
means = np.empty((n_stim_mc, N_mc))
for k in range(n_stim_mc):
    samp = simulate_ou(W_mc, S_mc, Sigma_eta_mc, cs[k], dt_mc, n_steps_mc,
                       n_trials_mc, burn_mc, rng_mc)
    flat = samp.reshape(-1, N_mc)
    means[k] = flat.mean(0)
    cond_cov += np.cov(flat.T) / n_stim_mc
marg_cov = cond_cov + np.cov(means.T)
I_mc = 0.5 * (np.linalg.slogdet(marg_cov)[1] - np.linalg.slogdet(cond_cov)[1])
mc_secs = time.time() - t0

rel_cond = float(np.linalg.norm(cond_cov - Src_theory) / np.linalg.norm(Src_theory))
rel_marg = float(np.linalg.norm(marg_cov - Sr_theory) / np.linalg.norm(Sr_theory))
mean_err = float(np.max(np.abs(means - cs @ (-Winv_S).T).mean(0)))

print(f"Monte Carlo: {n_stim_mc} stimuli x {n_trials_mc} trials x "
      f"{n_steps_mc - burn_mc} retained steps, dt = {dt_mc}  ({mc_secs:.0f} s)")
print()
check("conditional covariance, relative error", rel_cond, REF_rel_cond,
      tol=0.5, fmt="{:.4f}", key="mc_rel_cond")
check("marginal covariance, relative error", rel_marg, REF_rel_marg,
      tol=0.5, fmt="{:.4f}", key="mc_rel_marg")
check("closed-form MI, nats", I_theory, REF_I_theory, fmt="{:.4f}", key="mc_I_theory")
check("Monte-Carlo MI, nats", I_mc, REF_I_mc, tol=0.05, fmt="{:.4f}", key="mc_I_mc")
check("MI discrepancy, nats", abs(I_mc - I_theory), REF_I_gap,
      tol=1.0, fmt="{:.4f}", key="mc_I_gap")
for k_, v_ in (("dt_mc", dt_mc), ("n_steps_mc", n_steps_mc), ("n_trials_mc", n_trials_mc),
               ("n_stim_mc", n_stim_mc), ("N_mc", N_mc), ("M_mc", M_mc)):
    record(k_, v_)
'''

MD_FIG1 = r"""
## 3. Fig 1. The framework and its three limits

A schematic, drawn rather than computed. Panel (a) shows where the two noise conventions
inject their noise: Model A inside the recurrent loop, where the recurrence filters it
along with the signal, and Model B after the loop at a readout, where it does not. Panel
(b) shows the three published models the general formula contains, each as a setting of
the recurrent matrix, the sensitivity matrix, and the two dimensions.
"""
