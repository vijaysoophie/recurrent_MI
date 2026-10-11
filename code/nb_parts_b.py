"""Cell sources, part B: Fig 1 drawing, the two-noise sweep, Fig 2."""

CODE_FIG1 = r'''
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

def _box(ax, x, y, w, h, label, fc="white", ec=C_INK, fs=8.5, lw=1.0, tc="#111111"):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.03",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4, color=tc)

def _arrow(ax, p0, p1, color=C_INK, lw=1.1, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9,
                                 linewidth=lw, color=color, zorder=2, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))

fig = plt.figure(figsize=(fig_width(1.0), 3.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.12)

# ---------------------------------------------------------------- (a) the circuit
axa = fig.add_subplot(gs[0, 0]); axa.set_axis_off()
axa.set_xlim(0, 1); axa.set_ylim(0, 1)

_box(axa, 0.13, 0.62, 0.17, 0.115, "odor\n$\\mathbf{c}$", fc="#eef3f8", ec=C_MAIN)
_box(axa, 0.47, 0.62, 0.20, 0.115, "mitral\n$\\mathbf{M}$", fc="white", ec=C_INK)
_box(axa, 0.47, 0.25, 0.20, 0.115, "granule\n$\\mathbf{G}$", fc="#f4f4f4", ec=C_CTRL)
_box(axa, 0.84, 0.62, 0.17, 0.115, "readout\n$\\mathbf{y}$", fc="#fdeeee", ec=C_TARGET)

_arrow(axa, (0.215, 0.62), (0.375, 0.62))
_arrow(axa, (0.565, 0.62), (0.755, 0.62))
# reciprocal loop
_arrow(axa, (0.425, 0.565), (0.425, 0.305), color=C_INK, rad=0.35)
_arrow(axa, (0.515, 0.305), (0.515, 0.565), color=C_TARGET, rad=0.35)
axa.text(0.335, 0.435, "excite\n$\\mathbf{W}^{(gm)}$", ha="center", va="center",
         fontsize=7.0, color=C_INK)
axa.text(0.635, 0.435, "inhibit\n$w\\,\\mathbf{W}^{(gm)\\top}$", ha="center", va="center",
         fontsize=7.0, color=C_TARGET)

# noise injections
_arrow(axa, (0.47, 0.90), (0.47, 0.685), color=C_MAIN, lw=1.2)
axa.text(0.47, 0.935, "in-loop noise  $\\boldsymbol{\\eta}$", ha="center", va="bottom",
         fontsize=7.6, color=C_MAIN)
axa.text(0.47, 0.80, "Model A", ha="left", va="center", fontsize=7.2, color=C_MAIN,
         style="italic")

_arrow(axa, (0.84, 0.90), (0.84, 0.685), color=C_TARGET, lw=1.2)
axa.text(0.90, 0.935, "readout noise  $\\boldsymbol{\\nu}$", ha="center", va="bottom",
         fontsize=7.6, color=C_TARGET)
axa.text(0.855, 0.80, "Model B", ha="left", va="center", fontsize=7.2, color=C_TARGET,
         style="italic")

axa.text(0.69, 0.545, "gain $g$", ha="center", va="top", fontsize=7.0, color=C_INK)
axa.text(0.5, 0.055,
         "Model A: the loop filters signal and noise alike.\n"
         "Model B: the loop filters only the signal, and a\n"
         "gain holds total output power fixed.",
         ha="center", va="bottom", fontsize=7.3, color="#444444")
panel(axa, "a", dx=-4, dy=-6)

# ---------------------------------------------------------------- (b) the three limits
axb = fig.add_subplot(gs[0, 1]); axb.set_axis_off()
axb.set_xlim(0, 1); axb.set_ylim(0, 1)

axb.text(0.5, 0.965, "$I(\\mathbf{c},\\mathbf{r})=\\frac{1}{2}\\log\\det"
                     "[\\mathbf{I}+\\Sigma_{r|c}^{-1}\\mathbf{W}^{-1}\\mathbf{S}"
                     "\\Sigma_c\\mathbf{S}^{\\top}\\mathbf{W}^{-\\top}]$",
         ha="center", va="top", fontsize=8.6, color="#111111")

rows = [
    (0.735, "epithelium", "$\\mathbf{W}=-\\mathbf{I}$,  $\\mathbf{S}=\\mathbf{K}^{1/2}\\mathbf{A}$",
     "receptor abundances", C_CTRL),
    (0.495, "single neuron", "$N=1$,  $M=2$,  $\\mathbf{S}=(S_1,S_2)$",
     "receptor coexpression", C_CTRL),
    (0.255, "bulb", "$\\mathbf{W}=-\\mathbf{W}_{\\mathrm{eff}}$,  $\\mathbf{S}=\\mathbf{I}$",
     "recurrent decorrelation", C_MAIN),
]
for y, name, setting, what, col in rows:
    _arrow(axb, (0.5, y + 0.145), (0.5, y + 0.098), color="#bbbbbb", lw=0.9)
    axb.add_patch(FancyBboxPatch((0.045, y - 0.095), 0.91, 0.19,
                                 boxstyle="round,pad=0.008,rounding_size=0.02",
                                 linewidth=1.0, edgecolor=col,
                                 facecolor="#fafcfe" if col == C_MAIN else "white",
                                 zorder=3))
    axb.text(0.085, y + 0.045, name, ha="left", va="center", fontsize=8.2,
             fontweight="bold", color=col, zorder=4)
    axb.text(0.085, y - 0.048, setting, ha="left", va="center", fontsize=7.6,
             color="#333333", zorder=4)
    axb.text(0.915, y, what, ha="right", va="center", fontsize=7.4,
             color="#666666", style="italic", zorder=4)

axb.text(0.5, 0.075, "feedforward, scalar and full-recurrent limits\n"
                     "of one expression",
         ha="center", va="bottom", fontsize=7.3, color="#444444")
panel(axb, "b", dx=-4, dy=-6)

save(fig, "Fig1_framework")
plt.show()
'''

MD_SWEEP = r"""
## 4. Recurrent inhibition under two noise models

The central sweep. We vary the inhibitory gain $w$ over four decades and evaluate the
information under both noise conventions, for random and for source-matched wiring. Three
things are being tested.

**The monotonicity result.** Under Model A the information should fall monotonically with
$w$ for every connectivity, with no interior optimum anywhere.

**The interior optimum.** Under Model B it should rise to an interior optimum, but only
when the wiring is matched to the odor sources.

**Robustness of the monotonicity result to the adiabatic reduction.** The reduced
mitral-only description is obtained by eliminating the granule cells at their fixed point.
That step could in principle be what produces the monotone decline. We therefore also
solve the *exact* two-population Lyapunov equation, without any elimination, under three
placements of the in-loop noise: in the mitral population only, in both populations, and in
the granule population only. If the result is an artifact of the reduction, at least one of
these should show an interior optimum.

The sweep is deliberately long: 121 values of $w$ spanning three and a half decades, at 200
independent connectivity seeds each, so that the seed-to-seed spread is resolved rather than
assumed small, and the exact two-population check is run at 5 seeds per condition on a
larger network than the reduced description would need.
"""

CODE_SWEEP = r'''
# ---------------------------------------------------------------- the w sweep
N_g_main, N_CONNECT_F1 = 200, 10
N_SEEDS = 200
N_W_LOG = 120
w_values = np.concatenate(([0.0], np.logspace(-4, -0.5, N_W_LOG)))

t0 = time.time()
miA   = np.zeros((len(w_values), N_SEEDS))
miB_r = np.zeros((len(w_values), N_SEEDS))
miB_m = np.zeros((len(w_values), N_SEEDS))
flat_r = np.zeros((len(w_values), N_SEEDS))
flat_m = np.zeros((len(w_values), N_SEEDS))

W_rand = [make_wiring(N_m, N_g_main, N_CONNECT_F1, "random", seed=sd)
          for sd in range(N_SEEDS)]
W_match = [make_wiring(N_m, N_g_main, N_CONNECT_F1, "matched", blocks, seed=sd)
           for sd in range(N_SEEDS)]
K_rand = [W.T @ W for W in W_rand]
K_match = [W.T @ W for W in W_match]

for k, w_ in enumerate(w_values):
    for sd in range(N_SEEDS):
        Er = np.eye(N_m) + w_ * K_rand[sd]
        Em = np.eye(N_m) + w_ * K_match[sd]
        miA[k, sd]   = mi_inloop(Sigma_c, Er, sigma_noise)
        miB_r[k, sd] = mi_budget(Sigma_c, Er, sigma_noise, power_budget)
        miB_m[k, sd] = mi_budget(Sigma_c, Em, sigma_noise, power_budget)
        flat_r[k, sd] = spectral_flatness(output_covariance(Sigma_c, Er))
        flat_m[k, sd] = spectral_flatness(output_covariance(Sigma_c, Em))
sweep_secs = time.time() - t0

miA_m, miA_s = miA.mean(1), miA.std(1, ddof=1) / np.sqrt(N_SEEDS)
miBr_m, miBr_s = miB_r.mean(1), miB_r.std(1, ddof=1) / np.sqrt(N_SEEDS)
miBm_m, miBm_s = miB_m.mean(1), miB_m.std(1, ddof=1) / np.sqrt(N_SEEDS)

# per-seed monotonicity of Model A: the strongest form of the check
dA = np.diff(miA, axis=0)
frac_monotone = float((dA <= 1e-12).mean())
worst_increase = float(dA.max())

i_best_m = int(np.argmax(miBm_m))
i_best_r = int(np.argmax(miBr_m))
w_star = float(w_values[i_best_m])
gain_match = 100 * (miBm_m[i_best_m] / I_ff - 1)
gain_rand  = 100 * (miBr_m[i_best_r] / I_ff - 1)
# per-seed optimum, so the optimum itself carries an uncertainty
w_star_seeds = w_values[np.argmax(miB_m, axis=0)]
gain_match_seeds = 100 * (miB_m.max(0) / I_ff - 1)

# where random wiring falls back below the feedforward baseline
below = np.where(miBr_m < I_ff)[0]
w_cross_rand = float(w_values[below[0]]) if len(below) else np.nan

print(f"sweep: {len(w_values)} values of w x {N_SEEDS} seeds x 2 wirings "
      f"({sweep_secs:.0f} s)")
print()
check("Model A at w=0, nats", miA_m[0], REF_miA0, fmt="{:.2f}", key="miA_w0")
check("Model A at w=0.1, nats",
      float(np.interp(0.1, w_values, miA_m)), REF_miA01, fmt="{:.2f}", key="miA_w01")
check("Model A at w=0.316, nats", miA_m[-1], REF_miAend, fmt="{:.2f}", key="miA_wend")
check("Model A steps that decrease (fraction)", frac_monotone, REF_frac_mono,
      tol=0.0, fmt="{:.4f}", key="frac_monotone")
check("Model A largest single step, nats", worst_increase, REF_worst_inc,
      tol=1.0, fmt="{:.2e}", key="worst_increase")
record("modelA_n_steps", dA.size)
record("modelA_n_increasing", int((dA > 0).sum()))
check("Model B matched, w*", w_star, REF_wstar, tol=0.35, fmt="{:.4f}", key="w_star")
check("Model B matched gain, %", gain_match, REF_gain_match, tol=0.15, fmt="{:.2f}",
      key="gain_match")
check("Model B random gain, %", gain_rand, REF_gain_rand, tol=0.5, fmt="{:.2f}",
      key="gain_rand")
print(f"  per-seed optimum w* = {np.median(w_star_seeds):.4f} "
      f"[{np.percentile(w_star_seeds,5):.4f}, {np.percentile(w_star_seeds,95):.4f}] "
      f"(5th-95th percentile over {N_SEEDS} seeds)")
print(f"  per-seed matched gain = {gain_match_seeds.mean():.2f} "
      f"+/- {gain_match_seeds.std(ddof=1):.2f} %")
record("gain_match_sd", gain_match_seeds.std(ddof=1))
record("w_star_lo", np.percentile(w_star_seeds, 5))
record("w_star_hi", np.percentile(w_star_seeds, 95))
record("w_cross_rand", w_cross_rand)
for k_, v_ in (("N_SEEDS", N_SEEDS), ("n_w", len(w_values)), ("n_w_log", N_W_LOG),
               ("N_g_main", N_g_main), ("n_connect_f1", N_CONNECT_F1)):
    record(k_, v_)
'''

CODE_EXACT2POP = r'''
# ---------------------------------------------------------------- exact two-population check
def mi_two_population(W_gm, w, sigma2, Sigma_c, noise_in="mitral"):
    """I(c, M) for the FULL two-population system, with no adiabatic elimination.

    Stacked state R = (M, G) obeys dR/dt = W_full R + K_inj c + eta(t). The mitral-only
    readout information is the log-determinant ratio of the mitral blocks of the marginal
    and conditional covariances. 'noise_in' places the in-loop noise in the mitral
    population, in both populations, or in the granule population alone."""
    N_mm, N_gg = W_gm.shape[1], W_gm.shape[0]
    W_full = np.block([[-np.eye(N_mm), -w * W_gm.T],
                       [W_gm,          -np.eye(N_gg)]])
    K_inj = np.vstack([np.eye(N_mm), np.zeros((N_gg, N_mm))])
    d = np.concatenate([np.ones(N_mm) if noise_in in ("mitral", "both") else np.zeros(N_mm),
                        np.ones(N_gg) if noise_in in ("granule", "both") else np.zeros(N_gg)])
    Seta = sigma2 * np.diag(d + 1e-9)           # tiny floor keeps the solve well posed
    Src = solve_continuous_lyapunov(W_full, -Seta)
    B = -np.linalg.solve(W_full, K_inj)
    SR = Src + B @ Sigma_c @ B.T
    mm = slice(0, N_mm)
    return 0.5 * (np.linalg.slogdet(SR[mm, mm])[1] - np.linalg.slogdet(Src[mm, mm])[1])

# a smaller network, so that 200 stacked Lyapunov solves per curve stay quick
N_m2, N_g2, NC2, N_SEEDS2 = 30, 90, 4, 5
blocks2 = np.repeat(np.arange(5), N_m2 // 5)
Sigma_c2 = generate_odor_covariance(N_m2, 5)
# w = 0 is excluded: with noise in the granule population alone the mitral readout is
# noiseless there and its information diverges, so all curves are referenced to the
# smallest nonzero inhibition instead.
w_grid2 = np.logspace(-4, -0.5, 80)

t0 = time.time()
exact2, exact2_seeds = {}, {}
for placement in ("mitral", "both", "granule"):
    acc = np.zeros((N_SEEDS2, len(w_grid2)))
    for sd in range(N_SEEDS2):
        Wg2 = make_wiring(N_m2, N_g2, NC2, "random", seed=3 + sd)
        acc[sd] = [mi_two_population(Wg2, w_, 0.09, Sigma_c2, placement) for w_ in w_grid2]
    exact2[placement], exact2_seeds[placement] = acc.mean(0), acc
acc = np.zeros((N_SEEDS2, len(w_grid2)))
for sd in range(N_SEEDS2):
    Wg2m = make_wiring(N_m2, N_g2, NC2, "matched", blocks2, seed=3 + sd)
    acc[sd] = [mi_two_population(Wg2m, w_, 0.09, Sigma_c2, "both") for w_ in w_grid2]
exact2["matched, both"], exact2_seeds["matched, both"] = acc.mean(0), acc
exact2_secs = time.time() - t0

print(f"exact two-population sweep: 4 conditions x {len(w_grid2)} values of w x "
      f"{N_SEEDS2} seeds, N_m = {N_m2}, N_g = {N_g2}  ({exact2_secs:.0f} s)")
print()
for kk, vv in exact2.items():
    dd = np.diff(vv)
    print(f"  {kk:<16s} I(0) = {vv[0]:7.3f} -> I(w_max) = {vv[-1]:7.3f} nats, "
          f"monotone decreasing: {bool((dd <= 1e-10).all())}, "
          f"largest step increase {dd.max():+.2e}")
    record(f"exact2_{kk.replace(', ','_')}_I0", vv[0])
    record(f"exact2_{kk.replace(', ','_')}_Iend", vv[-1])
    record(f"exact2_{kk.replace(', ','_')}_maxstep", dd.max())

# monotonicity checked per seed, not on the seed average, which is the stronger statement
per_seed_steps = np.concatenate([np.diff(a, axis=1).ravel() for a in exact2_seeds.values()])
n_increasing = int((per_seed_steps > 1e-10).sum())
all_monotone = n_increasing == 0
print(f"\n  per-seed check: {per_seed_steps.size} steps across "
      f"{len(exact2_seeds)} conditions x {N_SEEDS2} seeds; {n_increasing} increase.")
print(f"  largest single step, over every condition and seed: {per_seed_steps.max():+.2e} nats")
record("exact2_all_monotone", float(all_monotone))
record("exact2_n_steps", per_seed_steps.size)
record("exact2_n_increasing", n_increasing)
check("exact two-population, largest single step, nats", per_seed_steps.max(),
      REF_exact2_worst, tol=1.0, fmt="{:.2e}", key="exact2_worst_step")
record("n_w_grid2", len(w_grid2))
record("N_SEEDS2", N_SEEDS2)
record("N_m2", N_m2); record("N_g2", N_g2)
'''

CODE_DECORR = r'''
# ---------------------------------------------------------------- decorrelation
W_DECORR = 0.05

def normalized_steady_state(W_eff, S_in, Sigma_c):
    """Gain-normalized steady state; the spontaneous offset drops out of correlations."""
    F = np.linalg.inv(W_eff)
    g2 = np.trace(Sigma_c) / np.trace(F @ Sigma_c @ F.T)
    return np.sqrt(g2) * (F @ S_in)

order = np.argsort(odor_source)
odors_sorted, src_sorted = odors[:, order], odor_source[order]
mask_all = np.zeros((N_ODORS, N_ODORS), dtype=bool)
mask_all[np.triu_indices(N_ODORS, k=1)] = True
mask_same = (src_sorted[:, None] == src_sorted[None, :]) & mask_all
mask_diff = (~(src_sorted[:, None] == src_sorted[None, :])) & mask_all

C_in = np.corrcoef(odors_sorted.T)
ss_in, ds_in = float(C_in[mask_same].mean()), float(C_in[mask_diff].mean())

decorr = {"Random": [], "Matched": []}
for sd in range(N_SEEDS):
    for name, Wl in (("Random", W_rand), ("Matched", W_match)):
        E = np.eye(N_m) + W_DECORR * (Wl[sd].T @ Wl[sd])
        Cc = np.corrcoef(normalized_steady_state(E, odors_sorted, Sigma_c).T)
        decorr[name].append((Cc[mask_same].mean(), Cc[mask_diff].mean()))
decorr = {k: np.array(v) for k, v in decorr.items()}

print(f"decorrelation at w = {W_DECORR}, {N_ODORS} odors, {N_SEEDS} seeds")
print()
check("same-source input correlation", ss_in, REF_ss_in, fmt="{:.3f}", key="ss_in")
check("output corr., random wiring", decorr["Random"][:, 0].mean(), REF_ss_rand,
      fmt="{:.3f}", key="ss_out_random")
check("output corr., matched wiring", decorr["Matched"][:, 0].mean(), REF_ss_match,
      tol=0.10, fmt="{:.3f}", key="ss_out_matched")
record("ss_out_matched_sd", decorr["Matched"][:, 0].std(ddof=1))
record("ss_out_random_sd", decorr["Random"][:, 0].std(ddof=1))
record("ds_in", ds_in)
record("ds_out_matched", decorr["Matched"][:, 1].mean())
'''

CODE_FIG2 = r'''
fig, axes = plt.subplots(1, 3, figsize=(fig_width(1.0), 2.5))
ax = axes.ravel()

# (a) Model A, reduced description
miA_match = np.array([[mi_inloop(Sigma_c, np.eye(N_m) + w_ * K_match[sd], sigma_noise)
                       for sd in range(N_SEEDS)] for w_ in w_values]).mean(1)
ax[0].plot(w_values[1:], miA_match[1:], color=C_TARGET, label="matched wiring")
ax[0].plot(w_values[1:], miA_m[1:], color=C_CTRL, ls="--", label="random wiring")
ax[0].fill_between(w_values[1:], (miA_m - 1.96 * miA_s)[1:], (miA_m + 1.96 * miA_s)[1:],
                   color=C_CTRL, alpha=0.22, linewidth=0)
ax[0].axhline(miA_m[0], color=C_INK, ls=":", lw=1.0, label="no inhibition")
ax[0].set_xscale("log"); ax[0].set_xlabel("inhibitory gain $w$")
ax[0].set_ylabel("$I(\\mathbf{c},\\mathbf{M})$  (nats)")
ax[0].set_title("Model A: noise inside the loop", fontsize=8.4)
ax[0].legend(loc="lower left")

# (b) Model B
ax[1].plot(w_values[1:], miBm_m[1:], color=C_TARGET, label="matched wiring")
ax[1].fill_between(w_values[1:], (miBm_m - 1.96 * miBm_s)[1:], (miBm_m + 1.96 * miBm_s)[1:],
                   color=C_TARGET, alpha=0.22, linewidth=0)
ax[1].plot(w_values[1:], miBr_m[1:], color=C_CTRL, ls="--", label="random wiring")
ax[1].fill_between(w_values[1:], (miBr_m - 1.96 * miBr_s)[1:], (miBr_m + 1.96 * miBr_s)[1:],
                   color=C_CTRL, alpha=0.25, linewidth=0)
ax[1].axhline(I_ff, color=C_INK, ls=":", lw=1.0, label="no inhibition")
ax[1].plot([w_star], [miBm_m[i_best_m]], "o", color=C_TARGET, ms=4.5, zorder=5)
_y0, _y1 = ax[1].get_ylim()
ax[1].set_ylim(_y0, _y1 + 0.10 * (_y1 - _y0))
ax[1].annotate(f"$w^*={w_star:.3f}$", xy=(w_star, miBm_m[i_best_m]),
               xytext=(12, 9), textcoords="offset points", fontsize=7.2, color=C_TARGET)
ax[1].set_xscale("log"); ax[1].set_xlabel("inhibitory gain $w$")
ax[1].set_ylabel("$I(\\mathbf{c},\\mathbf{y})$  (nats)")
ax[1].set_title("Model B: readout noise and a budget", fontsize=8.4)
ax[1].legend(loc="lower left")

# (c) decorrelation
labels = ["input", "random", "matched"]
vals = [ss_in, decorr["Random"][:, 0].mean(), decorr["Matched"][:, 0].mean()]
errs = [0.0, decorr["Random"][:, 0].std(ddof=1), decorr["Matched"][:, 0].std(ddof=1)]
cols = [C_FAINT, C_CTRL, C_TARGET]
ax[2].bar(labels, vals, yerr=errs, color=cols, capsize=3, width=0.6,
          error_kw=dict(lw=0.9, ecolor="#555555"))
ax[2].axhline(ss_in, color=C_INK, ls=":", lw=1.0)
for i, v in enumerate(vals):
    ax[2].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.4)
ax[2].set_ylim(0, 1.12)
ax[2].set_ylabel("same-source correlation")
ax[2].set_title(f"pattern decorrelation at $w={W_DECORR}$", fontsize=8.4)
ax[2].grid(axis="x", visible=False)

panels(axes)
fig.tight_layout()
save(fig, "Fig2_two_noise_models")
plt.show()
'''
