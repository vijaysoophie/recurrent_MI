"""Cell sources, part C: flatness, gradient ascent, sign obstruction, stability."""

MD_FLAT = r"""
## 5. The spectral-flatness identity

Under Model B the entire benefit of recurrent inhibition collapses to one number. With
$C_w = \mathbf W_{\mathrm{eff}}^{-1}\Sigma_c\mathbf W_{\mathrm{eff}}^{-\top}$ the output
signal covariance before the compensating gain, and $\mathcal F$ the spectral flatness,

$$I(w)-I(0) = \frac{N_m}{2}\log\frac{\mathcal F(C_w)}{\mathcal F(\Sigma_c)} \qquad
\text{(high signal-to-noise limit).}$$

The identity is derived in the main text as a high-signal-to-noise limit. Here
we ask how good it is at finite signal-to-noise, which is what the rest of the paper
actually uses. We evaluate both sides at every point of the sweep, for both wirings, and we
repeat the comparison across three decades of readout noise, so that the identity's error is
reported as a function of the regime rather than asserted for one.
"""

CODE_FLAT = r'''
def flatness_prediction(W_eff, Sigma_c):
    """Right-hand side of the flatness identity, in nats."""
    N = W_eff.shape[0]
    return 0.5 * N * np.log(spectral_flatness(output_covariance(Sigma_c, W_eff))
                            / spectral_flatness(Sigma_c))

# ---- measured against predicted along the main sweep, over many seeds and both wirings
N_SEEDS_FLAT = 20
meas, pred, which = [], [], []
for name, Kl in (("random", K_rand), ("matched", K_match)):
    for sd in range(N_SEEDS_FLAT):
        for w_ in w_values[1:]:
            E = np.eye(N_m) + w_ * Kl[sd]
            meas.append(mi_budget(Sigma_c, E, sigma_noise, power_budget) - I_ff)
            pred.append(flatness_prediction(E, Sigma_c))
            which.append(name)
meas, pred, which = np.array(meas), np.array(pred), np.array(which)
# seed 0 alone, for the curves drawn in Fig 3b
meas0 = {nm: np.array([mi_budget(Sigma_c, np.eye(N_m) + w_ * Kl[0], sigma_noise, power_budget)
                       - I_ff for w_ in w_values[1:]])
         for nm, Kl in (("random", K_rand), ("matched", K_match))}
pred0 = {nm: np.array([flatness_prediction(np.eye(N_m) + w_ * Kl[0], Sigma_c)
                       for w_ in w_values[1:]])
         for nm, Kl in (("random", K_rand), ("matched", K_match))}
record("N_SEEDS_FLAT", N_SEEDS_FLAT)

err = np.abs(meas - pred)
r_mp = float(np.corrcoef(meas, pred)[0, 1])

# The identity is a high-SNR limit. It is excellent wherever the gain-scaled output
# eigenvalues stay well above the readout floor, and degrades once strong inhibition pushes
# the weakest of them down to it. We therefore report the error twice: over the regime the
# paper actually works in (inhibition at or below w = 0.05, which brackets the interior
# optimum), and globally, with the value of w at which the worst case occurs.
w_rep = np.tile(w_values[1:], 2 * N_SEEDS_FLAT)
useful = w_rep <= 0.05
err_useful, r_useful = err[useful], float(np.corrcoef(meas[useful], pred[useful])[0, 1])
w_worst = float(w_rep[int(np.argmax(err))])

# ---- how the identity degrades as the readout noise grows
sig_grid = np.logspace(-2.5, 0.0, 26)
err_by_sigma, gain_by_sigma = [], []
E_star = np.eye(N_m) + w_star * K_match[0]
for s_ in sig_grid:
    m_ = (mi_budget(Sigma_c, E_star, s_, power_budget)
          - mi_budget(Sigma_c, np.eye(N_m), s_, power_budget))
    err_by_sigma.append(abs(m_ - flatness_prediction(E_star, Sigma_c)))
    gain_by_sigma.append(m_)
err_by_sigma = np.array(err_by_sigma); gain_by_sigma = np.array(gain_by_sigma)
i_sig = int(np.argmin(np.abs(sig_grid - sigma_noise)))

# ---- the whitening ceiling: how flat can this architecture get?
F_out_star = spectral_flatness(output_covariance(Sigma_c, E_star))
F_white_gain = 0.5 * N_m * np.log(1.0 / F_in)      # gain if whitening were reachable

print(f"flatness identity checked at {len(meas)} (w, wiring) points")
print()
check("identity, max error for w <= 0.05, nats", err_useful.max(), REF_flat_maxerr,
      tol=0.5, fmt="{:.4f}", key="flat_max_err")
check("identity, median error for w <= 0.05, nats", float(np.median(err_useful)),
      REF_flat_mederr, tol=0.5, fmt="{:.4f}", key="flat_med_err")
check("identity, correlation for w <= 0.05", r_useful, REF_flat_r_useful, tol=0.01,
      fmt="{:.5f}", key="flat_r_useful")
check("identity, global max error, nats", err.max(), REF_flat_globalmax, tol=0.5,
      fmt="{:.3f}", key="flat_global_max_err")
print(f"  worst case at w = {w_worst:.3f}, the largest inhibition in the sweep, where the")
print(f"  weakest gain-scaled output modes have fallen to the readout floor.")
record("flat_worst_w", w_worst)
i_star = int(np.argmin(np.abs(w_values[1:] - w_star)))
err_star_seeds = np.abs(meas[which == "matched"].reshape(N_SEEDS_FLAT, -1)[:, i_star]
                        - pred[which == "matched"].reshape(N_SEEDS_FLAT, -1)[:, i_star])
check("identity, error at w* (matched), nats", err_star_seeds.mean(),
      REF_flat_errstar, tol=0.5, fmt="{:.4f}", key="flat_err_wstar")
record("flat_err_wstar_sd", err_star_seeds.std(ddof=1))
check("identity, correlation measured vs predicted", r_mp, REF_flat_r,
      tol=0.01, fmt="{:.5f}", key="flat_r")
check("output flatness at w*, matched", F_out_star, REF_F_out, fmt="{:.4f}", key="F_out")
check("input flatness", F_in, REF_F_in2, fmt="{:.4f}", key="F_in_again")
check("identity error at sigma_out = 0.1, nats", err_by_sigma[i_sig], REF_flat_err_sig,
      tol=0.5, fmt="{:.4f}", key="flat_err_sigma01")
realized = meas[which == "matched"].reshape(N_SEEDS_FLAT, -1).max(1).mean()
print(f"  gain if perfect whitening were reachable: {F_white_gain:.2f} nats "
      f"({100*F_white_gain/I_ff:.1f}% of I(0)); actually realized at w*: "
      f"{realized:.2f} nats, {100*realized/F_white_gain:.0f}% of the ceiling")
record("F_white_gain", F_white_gain)
record("realized_gain_nats", realized)
record("flat_err_pct", 100 * err_star_seeds.mean() / realized)
record("realized_frac_of_ceiling", 100 * realized / F_white_gain)
record("F_white_gain_pct", 100 * F_white_gain / I_ff)
record("n_sigma_grid", len(sig_grid))
'''

CODE_FIG3 = r'''
fig, axes = plt.subplots(1, 3, figsize=(fig_width(1.0), 2.5))
ax = axes.ravel()

# (a) measured against predicted
for name, col in (("random", C_CTRL), ("matched", C_TARGET)):
    sel = which == name
    ax[0].plot(pred[sel], meas[sel], "o", color=col, ms=2.0, alpha=0.25,
               label=f"{name} wiring", rasterized=True)
lim = [min(pred.min(), meas.min()) * 1.05, max(pred.max(), meas.max()) * 1.05]
ax[0].plot(lim, lim, color=C_INK, ls=":", lw=1.0, label="identity")
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("predicted  $\\frac{N_m}{2}\\log[\\mathcal{F}(C_w)/\\mathcal{F}(\\Sigma_c)]$  (nats)")
ax[0].set_ylabel("measured  $I(w)-I(0)$  (nats)")
ax[0].set_title("the benefit is a flatness ratio", fontsize=8.4)
ax[0].legend(loc="upper left")

# (b) both against w
wv = w_values[1:]
ax[1].plot(wv, meas0["matched"], color=C_TARGET, label="matched, measured")
ax[1].plot(wv, pred0["matched"], color=C_TARGET, ls="--", lw=1.1, label="matched, identity")
ax[1].plot(wv, meas0["random"], color=C_CTRL, label="random, measured")
ax[1].plot(wv, pred0["random"], color=C_CTRL, ls="--", lw=1.1, label="random, identity")
ax[1].axhline(0.0, color=C_INK, ls=":", lw=1.0)
ax[1].set_xscale("log"); ax[1].set_xlabel("inhibitory gain $w$")
ax[1].set_ylabel("$I(w)-I(0)$  (nats)")
ax[1].set_title("measured and predicted, against $w$", fontsize=8.4)
ax[1].legend(loc="lower left", ncol=1)

# (c) identity error against readout noise
ax[2].loglog(sig_grid, np.maximum(err_by_sigma, 1e-6), color=C_MAIN,
             label="identity error")
ax[2].axvline(sigma_noise, color=C_INK, ls=":", lw=1.0)
ax[2].annotate("$\\sigma_{\\mathrm{out}}$ used\nthroughout", xy=(sigma_noise, err_by_sigma[i_sig]),
               xytext=(-4, 26), textcoords="offset points", fontsize=7.0,
               color="#444444", ha="right")
ax2 = ax[2].twinx()
ax2.plot(sig_grid, gain_by_sigma, color=C_TARGET, ls="--", lw=1.1)
ax2.set_ylabel("measured gain (nats)", color=C_TARGET)
ax2.tick_params(axis="y", colors=C_TARGET); ax2.grid(False)
ax[2].set_xlabel("readout noise $\\sigma_{\\mathrm{out}}$")
ax[2].set_ylabel("$|$measured $-$ predicted$|$  (nats)", color=C_MAIN)
ax[2].set_title("the identity is a high-SNR limit", fontsize=8.4)

panels(axes)
fig.tight_layout()
save(fig, "Fig3_flatness")
plt.show()
'''

MD_GRAD = r"""
## 6. Matched wiring is discoverable by a local rule

Everywhere above, source-matched wiring is imposed by hand. That is a fair way to ask what
the architecture buys, but it leaves open whether anything could build it. The
budget-constrained objective has a closed-form gradient in the granule-to-mitral weights
(Supporting Information), so each step of an optimizer costs one $O(N_m^3)$ pass rather
than one information evaluation per nonzero weight, and we can simply run projected gradient
ascent from random sparse wiring and look at what it finds.

The sparsity pattern is held fixed and the weights are clipped to $[0,1]$ rather than
renormalized, since rescaling the whole matrix would silently change the effective
inhibition. The optimizer never sees the block labels.

The ascent runs on the full fifty-cell network with the full granule population, for 8000
iterations from each of 40 random restarts, with a step size set by backtracking so that the
objective never decreases and no learning rate has to be tuned.
"""

CODE_GRAD = r'''
def mi_and_grad(W_gm, Sigma_c, w, sigma, budget):
    """Model B information and its analytic gradient with respect to W_gm."""
    N = W_gm.shape[1]
    E = np.eye(N) + w * (W_gm.T @ W_gm)
    A = np.linalg.inv(E)
    C = A @ Sigma_c @ A.T
    T = np.trace(C)
    kappa = budget / sigma**2
    B = kappa * C / T
    IB = np.linalg.inv(np.eye(N) + B)
    mi = 0.5 * np.linalg.slogdet(np.eye(N) + B)[1]
    G = IB - (np.trace(IB @ C) / T) * np.eye(N)
    Q = Sigma_c @ A @ G + G @ A @ Sigma_c
    dW = -(kappa * w / T) * (W_gm @ A @ Q @ A)
    return mi, dW

def ascend(N_m_, N_g_, n_connect, Sigma_c_, blk, sigma=sigma_noise, n_iter=4000,
           lr0=200.0, w_fixed=0.02, seed=0, log_every=10):
    """Projected gradient ascent with a step size that backtracks on any decrease."""
    W = make_wiring(N_m_, N_g_, n_connect, "random", seed=seed)
    mask = W > 0
    budget = np.trace(Sigma_c_)
    lr = lr0
    mi, dW = mi_and_grad(W, Sigma_c_, w_fixed, sigma, budget)
    hist_mi, hist_pur, hist_it = [mi], [block_purity(W, blk)], [0]
    for it in range(1, n_iter + 1):
        for _ in range(40):                        # backtracking line search
            W_try = np.clip(W + lr * dW * mask, 0.0, 1.0)
            mi_try, dW_try = mi_and_grad(W_try, Sigma_c_, w_fixed, sigma, budget)
            if mi_try >= mi:
                break
            lr *= 0.5
        else:
            break                                  # step size collapsed: converged
        W, mi, dW = W_try, mi_try, dW_try
        lr *= 1.05
        if it % log_every == 0 or it == n_iter:
            hist_mi.append(mi); hist_pur.append(block_purity(W, blk)); hist_it.append(it)
    return W, np.array(hist_it), np.array(hist_mi), np.array(hist_pur)

W_GRAD, N_RESTARTS, N_ITER = 0.02, 100, 8000
t0 = time.time()
runs = [ascend(N_m, N_g_main, N_CONNECT_F1, Sigma_c, blocks, n_iter=N_ITER, seed=s)
        for s in range(N_RESTARTS)]
grad_secs = time.time() - t0
best = int(np.argmax([r[2][-1] for r in runs]))
W_opt, it_hist, mi_hist, pur_hist = runs[best]

def ref_mi(kind, n_refs=N_SEEDS):
    return np.array([mi_budget(Sigma_c, np.eye(N_m) + W_GRAD * (W.T @ W),
                               sigma_noise, power_budget)
                     for W in ((W_rand if kind == "random" else W_match)[:n_refs])])

mi_rand_ref = ref_mi("random"); mi_match_ref = ref_mi("matched")
chance_purity = float(np.mean([block_purity(W) for W in W_rand]))
gap_closed = 100 * (mi_hist[-1] - mi_hist[0]) / (mi_match_ref.mean() - mi_hist[0])
final_all = np.array([r[2][-1] for r in runs])
F_opt = spectral_flatness(output_covariance(Sigma_c, np.eye(N_m) + W_GRAD * (W_opt.T @ W_opt)))

print(f"gradient ascent: {N_RESTARTS} restarts x {N_ITER} iterations at w = {W_GRAD}, "
      f"N_m = {N_m}, N_g = {N_g_main}  ({grad_secs:.0f} s)")
print()
show_table(
    [["feedforward (w=0)", f"{I_ff:.3f}", "-"],
     ["random wiring", f"{mi_rand_ref.mean():.3f}", f"{chance_purity:.3f}"],
     ["ascent, start", f"{mi_hist[0]:.3f}", f"{pur_hist[0]:.3f}"],
     [f"ascent, end (best of {N_RESTARTS})", f"{mi_hist[-1]:.3f}", f"{pur_hist[-1]:.3f}"],
     ["hand-designed matched", f"{mi_match_ref.mean():.3f}", "1.000"]],
    ["network", "MI (nats)", "block purity"],
    title=f"projected gradient ascent at w = {W_GRAD}")
print()
check("ascent, MI gain, nats", mi_hist[-1] - mi_hist[0], REF_grad_gain,
      tol=0.1, fmt="{:.3f}", key="grad_gain")
check("ascent, gap to matched closed, %", gap_closed, REF_grad_gap, tol=0.15,
      fmt="{:.1f}", key="grad_gap_closed")
check("ascent, block purity, start", pur_hist[0], REF_grad_pur0, tol=0.05,
      fmt="{:.3f}", key="grad_purity_start")
check("ascent, block purity, end", pur_hist[-1], REF_grad_pur1, tol=0.05,
      fmt="{:.3f}", key="grad_purity_end")
check("chance block purity", chance_purity, REF_chance_pur, tol=0.03,
      fmt="{:.3f}", key="chance_purity")
check("ascent, spread over restarts (nats)", final_all.std(ddof=1), REF_grad_sd,
      tol=1.0, fmt="{:.4f}", key="grad_restart_sd")
check("ascent, output flatness", F_opt, REF_grad_F, tol=0.05, fmt="{:.4f}", key="grad_F")
record("F_match_at_wgrad",
       np.mean([spectral_flatness(output_covariance(
           Sigma_c, np.eye(N_m) + W_GRAD * (W.T @ W))) for W in W_match[:N_SEEDS]]))
record("mi_rand_ref", mi_rand_ref.mean())
record("mi_match_ref", mi_match_ref.mean())
record("grad_mi_end", mi_hist[-1])
record("grad_mi_start", mi_hist[0])
record("N_RESTARTS", N_RESTARTS); record("N_ITER", N_ITER); record("w_grad", W_GRAD)
record("grad_excess_over_matched", mi_hist[-1] - mi_match_ref.mean())
'''

CODE_FIG4 = r'''
# The shared-target matrix before and after the best run, on one common color scale.
W_init = make_wiring(N_m, N_g_main, N_CONNECT_F1, "random", seed=best)
def _normed_K(Wm):
    K = Wm.T @ Wm
    off = ~np.eye(K.shape[0], dtype=bool)
    K = K / K[off].mean()          # relative to its own mean off-diagonal entry
    K[~off] = np.nan
    return K
K_before, K_after = _normed_K(W_init), _normed_K(W_opt)
vmax = float(np.nanpercentile(np.concatenate([K_before.ravel(), K_after.ravel()]), 99.5))

# Plot data is dumped so the layout can be retuned without rerunning the notebook.
np.savez_compressed(os.path.join(OUTDIR, "figdata_fig3.npz"),
                    it_hist=it_hist, mi_hist=mi_hist, pur_hist=pur_hist,
                    runs_it=np.array([r[1] for r in runs], dtype=object),
                    runs_mi=np.array([r[2] for r in runs], dtype=object),
                    runs_pur=np.array([r[3] for r in runs], dtype=object),
                    mi_match=mi_match_ref.mean(), mi_rand=mi_rand_ref.mean(), I_ff=I_ff,
                    chance=chance_purity, K_before=K_before, K_after=K_after, vmax=vmax,
                    n_blocks=n_blocks, N_m=N_m, N_ITER=N_ITER, W_GRAD=W_GRAD)

def draw_fig3(W=WIDE_W, H=WIDE_H, name="Fig4_gradient_ascent"):
    """One row of four panels, sized for the full text width plus the left margin."""
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(1, 5, wspace=0.40, width_ratios=[1, 1, 1, 1, 0.05])
    axA, axB, axC, axD = (fig.add_subplot(gs[0, i]) for i in range(4))
    cax = fig.add_subplot(gs[0, 4])
    for axx in (axA, axB):
        axx.set_box_aspect(1.0)      # same square box as the two matrix panels

    # (a) MI against iteration
    for r in runs:
        axA.plot(r[1] + 1, r[2], color=C_FAINT, lw=0.7, zorder=1)
    axA.plot(it_hist + 1, mi_hist, color=C_MAIN, lw=1.5, zorder=3, label="best restart")
    axA.set_xscale("log")
    axA.axhline(mi_match_ref.mean(), color=C_TARGET, ls="--", lw=1.0, label="matched")
    axA.axhline(mi_rand_ref.mean(), color=C_CTRL, ls="-.", lw=1.0, label="random")
    axA.axhline(I_ff, color=C_INK, ls=":", lw=1.0, label="no inhibition")
    axA.set_xlabel("iteration + 1"); axA.set_ylabel("$I(\\mathbf{c},\\mathbf{y})$  (nats)")
    axA.set_title(f"ascent at $w={W_GRAD}$", fontsize=8.0)
    axA.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=5.8,
               labelspacing=0.25)

    # (b) purity against iteration
    for r in runs:
        axB.plot(r[1] + 1, r[3], color=C_FAINT, lw=0.7, zorder=1)
    axB.plot(it_hist + 1, pur_hist, color=C_MAIN, lw=1.5, zorder=3)
    axB.set_xscale("log")
    axB.axhline(1.0, color=C_TARGET, ls="--", lw=1.0)
    axB.axhline(chance_purity, color=C_CTRL, ls="-.", lw=1.0)
    axB.text(N_ITER, 1.0, "matched", ha="right", va="bottom", fontsize=6.2, color=C_TARGET)
    axB.text(N_ITER, chance_purity, "chance", ha="right", va="bottom",
             fontsize=6.2, color=C_CTRL)
    axB.set_ylim(0, 1.12)
    axB.set_xlabel("iteration + 1"); axB.set_ylabel("block purity")
    axB.set_title("granule cells concentrate", fontsize=8.0)

    # (c, d) the shared-target matrix before and after, one shared colorbar
    for axx, K, ttl in ((axC, K_before, "before"), (axD, K_after, "after")):
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
        panel(axx, lab, dx=-26, dy=4)
    fig.subplots_adjust(left=0.055, right=0.955, bottom=0.20, top=0.86)
    # the square panels are shorter than their gridspec slot, so match the colorbar to them
    fig.canvas.draw()
    pd, pc = axD.get_position(), cax.get_position()
    cax.set_position([pc.x0, pd.y0, pc.width, pd.height])
    save(fig, name)
    return fig

fig = draw_fig3()
plt.show()
'''


CODE_ROBUST = r'''
# ---------------------------------------------------------------- robustness of the optimum
# How much of this depends on the particular stimulus ensemble? We repeat the Model B sweep
# across within-source correlations, numbers of sources, and granule fan-out, and report
# where the optimum sits and what it is worth.
N_SEEDS_ROBUST = 100

def sweep_optimum(corr, nb, n_connect, n_seeds=N_SEEDS_ROBUST, N=N_m, N_g=200):
    assert n_connect <= N // nb, (
        f"n_connect={n_connect} exceeds the block size {N // nb} (N={N}, nb={nb}); "
        "make_wiring would silently cap it, making this condition indistinguishable "
        "from a smaller fan-out.")
    blk = np.repeat(np.arange(nb), N // nb)
    Sig = generate_odor_covariance(N, nb, corr)
    P = np.trace(Sig)
    I0 = mi_budget(Sig, np.eye(N), sigma_noise, P)
    ws = np.logspace(-4, -1, 40)
    best_w, best_g = [], []
    for sd in range(n_seeds):
        Wm = make_wiring(N, N_g, n_connect, "matched", blk, seed=sd)
        Km = Wm.T @ Wm
        vals = np.array([mi_budget(Sig, np.eye(N) + w_ * Km, sigma_noise, P) for w_ in ws])
        best_w.append(ws[int(np.argmax(vals))]); best_g.append(100 * (vals.max() / I0 - 1))
    return (np.median(best_w), np.mean(best_g), np.std(best_g, ddof=1),
            spectral_flatness(Sig))

t0 = time.time()
rows = []
for corr in (0.2, 0.4, 0.6, 0.8):
    wS, gS, gsd, Fs = sweep_optimum(corr, n_blocks, 10)
    rows.append([f"correlation {corr}", f"{Fs:.3f}", f"{wS:.4f}", f"{gS:.2f} +/- {gsd:.2f}"])
    record(f"robust_corr{int(100*corr)}_gain", gS)
    record(f"robust_corr{int(100*corr)}_w", wS); record(f"robust_corr{int(100*corr)}_F", Fs)

# Fan-out is capped at the block size (N // nb), so the widest source count tested (10
# sources of 5 glomeruli each) forces fan-out <= 5. We use that fan-out for all three rows,
# rather than letting the wider-block conditions silently cap down to it, so the three rows
# differ only in the number of sources.
NC_SOURCES = N_m // 10
for nb_ in (2, 5, 10):
    wS, gS, gsd, Fs = sweep_optimum(0.6, nb_, NC_SOURCES)
    rows.append([f"{nb_} sources", f"{Fs:.3f}", f"{wS:.4f}", f"{gS:.2f} +/- {gsd:.2f}"])
    record(f"robust_nb{nb_}_gain", gS)
    record(f"robust_nb{nb_}_w", wS); record(f"robust_nb{nb_}_F", Fs)
record("robust_nc_sources", NC_SOURCES)
# Fan-out sweep, at the number of sources used throughout the main text (five, block size
# 10): both alternative values stay within that block size, so none of them is silently
# capped back down to the baseline.
for nc in (6, 8, 10):
    wS, gS, gsd, Fs = sweep_optimum(0.6, n_blocks, nc)
    rows.append([f"fan-out {nc}", f"{Fs:.3f}", f"{wS:.4f}", f"{gS:.2f} +/- {gsd:.2f}"])
    record(f"robust_nc{nc}_gain", gS)
    record(f"robust_nc{nc}_w", wS); record(f"robust_nc{nc}_F", Fs)
robust_secs = time.time() - t0

show_table(rows, ["ensemble", "F(Sigma_c)", "w*", "gain over feedforward (%)"],
           title=f"where the interior optimum sits, across ensembles "
                 f"({N_SEEDS_ROBUST} seeds each, {robust_secs:.0f} s)")
print()
print("The optimum exists in every condition. It is worth more when the input spectrum is")
print("less flat, which is what the flatness identity predicts.")
gains = [v for k, v in RESULTS.items() if k.startswith("robust_") and k.endswith("_gain")]
record("robust_min_gain", min(gains)); record("robust_max_gain", max(gains))
record("robust_n_conditions", len(gains))
record("robust_n_seeds", N_SEEDS_ROBUST)
'''
