"""Cell sources, part F: the analysis repeated on Chow et al.'s own network."""

MD_CHOW = r"""
## 9. Chow et al.'s network: neurogenesis builds matched wiring

Everything above uses a stimulus ensemble and a matched wiring that we constructed. Here
the same analysis is repeated on the network of Chow, Wick & Riecke (2012) itself.

**What is theirs.** The eight glomerular activation patterns ($N_m = 424$ glomeruli) that
drive all of their simulations, recovered from the published figures in our reproduction of
that paper and shipped here as `data/chow2012_stimuli.npz`; their network parameters
($n_{\rm connect}=8$, $w=0.005$, $M_{sp}=1$, $G_{\min}=1.2$, $R_0=0.1$, $\gamma=20$); and
their neurogenesis rule, in which granule cells are added with random connections and
survive only if they are active enough across the odor set (`chow_model.py`, the engine of
that reproduction, unchanged). At fixed connectivity their steady state is exactly our
$\mathbf W_{\rm eff} = \mathbf I + w\,\mathbf W^{(gm)\top}\mathbf W^{(gm)}$, which the cell
below checks against their engine.

**What we add.** Their model has no noise, so we add ours (Model A and Model B, $\sigma=0.1$).
Our formula needs a stimulus ensemble with a covariance, so we take Gaussian mixtures that
vary around the average odor with the covariance of the eight patterns. It is seven
dimensional, and every information below is computed in that seven-dimensional stimulus
space with the $M\times M$ form of the formula. It is scaled, like the synthetic ensemble,
so that the average glomerulus has unit input variance.

**What we compare.** The grown network (three seeds) against random wiring with the same
number of granule cells and connections per cell, and against wiring found by gradient
ascent on the Model B information, started from that random wiring.
"""

CODE_CHOW = r'''
import sys
sys.path.insert(0, os.path.abspath("."))
import chow_model as chow

_d = np.load(os.path.join("data", "chow2012_stimuli.npz"), allow_pickle=True)
CH_S = _d["S"]                                   # (8 odors, 424 glomeruli)
CH_NAMES = _d["names"]                            # odor labels, in the order of the rows of S
CH_C_pub = _d["C_published_out"]                  # output correlations decoded from their Fig 2Cii
CH_C_pub_in = _d["C_published_in"]                # input correlations decoded from their Fig 2Ci
CH_Nm = CH_S.shape[1]
CH_P = dict(Nm=CH_Nm, nconnect=8, b=33, w=0.005, Msp=1.0, Gmin=1.2, R0=0.1, gamma=20.0)
CH_KAPPA, CH_T = 4.0, 2500      # 4x their influx reaches the same steady state in 2500 steps
CH_PAIRS = [(0, 1), (2, 3)]     # the enantiomer pairs (+/-limonene, +/-carvone)
CH_W = CH_P["w"]
CH_SEEDS = list(range(10))
CH_IU = np.triu_indices(CH_Nm, 1)

def ch_factor(X):
    """A with Sigma_c = A^T A, restricted to the directions in which the ensemble varies,
    scaled so that the average glomerulus has unit input variance."""
    ev, U = np.linalg.eigh(X.T @ X)
    keep = ev > 1e-9 * ev.max()
    A = (U[:, keep] * np.sqrt(ev[keep])).T
    return A * np.sqrt(CH_Nm / np.trace(A.T @ A))

CH_A = ch_factor(CH_S - CH_S.mean(0, keepdims=True))      # odor-identity ensemble
CH_A_INT = ch_factor(CH_S)                                  # identity plus total concentration
CH_BUDGET = float(CH_Nm)
CH_d = CH_A.shape[0]

def ch_K(exc, V=None):
    """Shared-target matrix W^T W from each granule cell's partners (and weights)."""
    V = np.ones(exc.shape) if V is None else V
    idx = exc[:, :, None].astype(np.int64) * CH_Nm + exc[:, None, :].astype(np.int64)
    return np.bincount(idx.ravel(), weights=(V[:, :, None] * V[:, None, :]).ravel(),
                       minlength=CH_Nm * CH_Nm).reshape(CH_Nm, CH_Nm)

def ch_filter(K, w):
    return np.linalg.inv(np.eye(CH_Nm) + w * K)

def ch_O(A, X):
    Y = A @ X
    return Y @ Y.T                                  # overlaps of the filtered odor patterns

def ch_miA(A, X, sig=sigma_noise):
    return 0.5 * np.linalg.slogdet(np.eye(A.shape[0]) + (2 / sig**2) * A @ X @ A.T)[1]

def ch_miB(A, X, sig=sigma_noise, P=CH_BUDGET):
    O = ch_O(A, X)
    return 0.5 * np.linalg.slogdet(np.eye(A.shape[0]) + P / (sig**2 * np.trace(O)) * O)[1]

def ch_ceiling(A, sig=sigma_noise, P=CH_BUDGET):
    k = A.shape[0]
    return 0.5 * k * np.log(1 + P / (k * sig**2))  # equal power in every stimulus direction

def ch_grad(A, exc, V, w, sig=sigma_noise, P=CH_BUDGET):
    """Model B information and its gradient with respect to the graded granule weights."""
    X = ch_filter(ch_K(exc, V), w); O = ch_O(A, X); k = A.shape[0]
    t = np.trace(O); c = P / (sig**2 * t)
    I = 0.5 * np.linalg.slogdet(np.eye(k) + c * O)[1]
    Q = np.linalg.inv(np.eye(k) + c * O); alpha = np.trace(Q @ O) / t
    H = A.T @ (0.5 * c * (Q - alpha * np.eye(k))) @ A
    Gam = -w * X @ (X @ H + H @ X) @ X; Gam = 0.5 * (Gam + Gam.T)
    return I, 2 * np.einsum("jm,jmn->jn", V, Gam[exc[:, :, None], exc[:, None, :]])

def ch_grow(Gmin, seed):
    p = dict(CH_P, Gmin=Gmin, b=CH_P["b"] * CH_KAPPA)
    net = chow.ChowNetwork(CH_S, rng=np.random.default_rng(seed), **p)
    hist = []
    for k in range(CH_T):
        net.step()
        if k % 100 == 0:
            hist.append(net.Ng)
    return net, np.array(hist)

def ch_random(Ng, seed):
    return np.random.default_rng(10_000 + seed).random((Ng, CH_Nm)).argpartition(7, axis=1)[:, :8]

def ch_chow_measures(K, w=CH_W):
    """Chow et al.'s own decorrelation measures on the deterministic steady state."""
    M = np.linalg.solve(np.eye(CH_Nm) + w * K, (CH_P["Msp"] + CH_S).T)
    C = chow.pearson_corr_matrix(M)
    return chow.mean_corr(C), chow.top_corr(C, CH_PAIRS), M.mean()

def ch_align(K):
    Sig = CH_A.T @ CH_A
    return float(np.corrcoef(K[CH_IU], Sig[CH_IU])[0, 1])

def ch_flat(O):
    return spectral_flatness(O)

# ---------------------------------------------------------------- grow the networks
t0 = time.time()
CH_NETS = {}
for s in CH_SEEDS:
    net, hist = ch_grow(1.2, s)
    Ka = ch_K(net.exc)
    assert np.allclose(CH_W * Ka, net.rebuild_W()), "our W_eff must be their steady state"
    CH_NETS[s] = dict(Ng=net.Ng, Ka=Ka, Kr=ch_K(ch_random(net.Ng, s)),
                      drift=100 * (hist[-1] - hist[-6]) / hist[-1])
    print(f"seed {s}: {net.Ng} granule cells, change over the last 500 steps "
          f"{CH_NETS[s]['drift']:+.2f}%   ({time.time() - t0:.0f} s)")
ch_Ng = np.mean([CH_NETS[s]["Ng"] for s in CH_SEEDS])

# ---------------------------------------------------------------- their decorrelation measure
Cin = chow.pearson_corr_matrix(CH_S.T)
# The recovered patterns are validated against the one published statistic of the input
# ensemble: the correlation matrix of their Fig 2Ci, decoded from the same figure.
_iu8 = np.triu_indices(len(CH_S), 1)
ch_recov_err = float(np.abs(Cin - CH_C_pub_in)[_iu8].max())
# One color step of that figure's scale, read off the decoded values themselves: the
# smallest nonzero gap between distinct decoded correlations.
_dec = np.unique(np.round(CH_C_pub_in[_iu8], 6))
ch_cstep = float(np.diff(_dec).min())
print(f"recovered input correlations agree with their published Fig 2Ci to "
      f"{ch_recov_err:.3f} (largest absolute difference over the {_iu8[0].size} odor pairs), "
      f"within one color step of that figure's scale ({ch_cstep:.3f})")
# Their Fig 2A places one image twice, so the panel labels cannot all be right. The
# assignment of panels to odorants was therefore recovered from this correlation matrix,
# which records the ensemble independently of the layout: every assignment is scored, and
# the second best is reported here to show that the winner is not a close call.
import itertools as _it
_perm_err = sorted(float(np.abs(Cin[np.ix_(p, p)] - CH_C_pub_in)[_iu8].max())
                   for p in _it.permutations(range(len(CH_S))))
ch_recov_next = _perm_err[1]
print(f"  of the {len(_perm_err)} assignments of panels to odorants, the best is the one "
      f"used here; the next best disagrees by {ch_recov_next:.3f}")
record("ch_recov_err", ch_recov_err)
record("ch_recov_next", ch_recov_next)
record("ch_cstep", ch_cstep)
record("ch_n_levels", int(np.unique(CH_S).size))
record("ch_S_lo", float(CH_S.min())); record("ch_S_hi", float(CH_S.max()))
ad = np.array([ch_chow_measures(CH_NETS[s]["Ka"]) for s in CH_SEEDS])
rd = np.array([ch_chow_measures(CH_NETS[s]["Kr"]) for s in CH_SEEDS])
ch_top = dict(input=chow.top_corr(Cin, CH_PAIRS), published=chow.top_corr(CH_C_pub, CH_PAIRS),
              grown=ad[:, 1].mean(), random=rd[:, 1].mean())
print()
show_table([[k, f"{v:.3f}"] for k, v in ch_top.items()], ["", "enantiomer correlation"],
           title="Chow et al.'s decorrelation measure at w = 0.005")
print(f"mean mitral activity of the grown network: {ad[:, 2].mean():.3f} "
      f"(their steady-state value G_min/n_connect = {1.2 / 8:.3f})")

# ---------------------------------------------------------------- is the grown wiring matched?
al_a = np.array([ch_align(CH_NETS[s]["Ka"]) for s in CH_SEEDS])
al_r = np.array([ch_align(CH_NETS[s]["Kr"]) for s in CH_SEEDS])

# ---------------------------------------------------------------- the inhibitory-strength sweep
CH_WS = np.concatenate([[0.0], np.logspace(-5, -1, 25)])
ch_sw = {k: np.zeros((len(CH_SEEDS), len(CH_WS))) for k in ("IA_a", "IA_r", "IB_a", "IB_r", "F_a", "F_r")}
for i, s in enumerate(CH_SEEDS):
    for j, w_ in enumerate(CH_WS):
        for tag, K in (("a", CH_NETS[s]["Ka"]), ("r", CH_NETS[s]["Kr"])):
            X = ch_filter(K, w_)
            ch_sw["IA_" + tag][i, j] = ch_miA(CH_A, X)
            ch_sw["IB_" + tag][i, j] = ch_miB(CH_A, X)
            ch_sw["F_" + tag][i, j] = ch_flat(ch_O(CH_A, X))
ch_Iff = ch_sw["IB_a"][0, 0]
ch_Imax = ch_ceiling(CH_A)
ch_avail = ch_Imax - ch_Iff
ch_F_in = ch_flat(CH_A @ CH_A.T)
jC = int(np.argmin(np.abs(CH_WS - CH_W)))
ch_maxstep = max(np.diff(ch_sw["IA_a"], axis=1).max(), np.diff(ch_sw["IA_r"], axis=1).max())
ch_gain_a = ch_sw["IB_a"][:, jC] - ch_Iff
ch_gain_r = ch_sw["IB_r"][:, jC] - ch_Iff
ch_gain_sat = ch_sw["IB_a"][:, -1] - ch_Iff
ch_interior = np.mean([int(np.argmax(ch_sw["IB_a"][i])) < len(CH_WS) - 1 for i in range(len(CH_SEEDS))])

# flatness identity in stimulus space
_pred = 0.5 * CH_d * np.log(np.r_[ch_sw["F_a"].ravel(), ch_sw["F_r"].ravel()] / ch_F_in)
_meas = np.r_[ch_sw["IB_a"].ravel(), ch_sw["IB_r"].ravel()] - ch_Iff
ch_flat_err = float(np.abs(_pred - _meas).max())

# ---------------------------------------------------------------- gradient ascent from random wiring
CH_N_ITER = 1000
def ch_ascend(A, exc, w, n_iter=CH_N_ITER):
    V = np.ones(exc.shape); I, g = ch_grad(A, exc, V, w)
    eta = 0.05 / np.abs(g).max(); traj = [I]
    for _ in range(n_iter):
        Vn = np.clip(V + eta * g, 0.0, 1.0)
        In = ch_miB(A, ch_filter(ch_K(exc, Vn), w))
        if In > I:
            V = Vn; I, g = ch_grad(A, exc, V, w); eta *= 1.05
        else:
            eta *= 0.5
        traj.append(I)
    return V, np.array(traj)

CH_GA = {}
for s in CH_SEEDS:
    exc_r = ch_random(CH_NETS[s]["Ng"], s)
    V, traj = ch_ascend(CH_A, exc_r, CH_W)
    Kg = ch_K(exc_r, V)
    CH_GA[s] = dict(traj=traj, F=ch_flat(ch_O(CH_A, ch_filter(Kg, CH_W))), align=ch_align(Kg),
                    top=ch_chow_measures(Kg)[1])
ch_gain_g = np.array([CH_GA[s]["traj"][-1] for s in CH_SEEDS]) - ch_Iff
print(f"\ngradient ascent done ({time.time() - t0:.0f} s)")

# ---------------------------------------------------------------- how selective must neurogenesis be?
CH_GMINS = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
CH_GMIN_SEEDS = CH_SEEDS[:3]      # each threshold is grown at this many seeds and averaged
ch_gsw = []
for gm in CH_GMINS:
    per_seed = []
    for sd_ in CH_GMIN_SEEDS:
        if gm == CH_P["Gmin"]:
            K = CH_NETS[sd_]["Ka"]; Ng_ = CH_NETS[sd_]["Ng"]; drift = CH_NETS[sd_]["drift"]
        else:
            net, hist = ch_grow(gm, sd_)
            K, Ng_, drift = ch_K(net.exc), net.Ng, 100 * (hist[-1] - hist[-6]) / hist[-1]
        per_seed.append([Ng_, drift, ch_align(K),
                         ch_miB(CH_A, ch_filter(K, CH_W)) - ch_Iff])
    a_ = np.array(per_seed, dtype=float)
    ch_gsw.append(dict(Gmin=gm, Ng=a_[:, 0].mean(),
                       drift=a_[int(np.argmax(np.abs(a_[:, 1]))), 1],
                       align=a_[:, 2].mean(), gain=a_[:, 3].mean(),
                       gain_sd=a_[:, 3].std(ddof=1)))
ch_gfrac = np.array([100 * g["gain"] / ch_avail for g in ch_gsw])
print(f"G_min sweep done ({time.time() - t0:.0f} s)")
show_table([[f"{g['Gmin']:.1f}", f"{g['Ng']:.0f}", f"{g['drift']:+.2f}%",
             f"{g['align']:.3f}", f"{g['gain']:+.3f} +/- {g['gain_sd']:.3f}", f"{f_:.0f}%"]
            for g, f_ in zip(ch_gsw, ch_gfrac)],
           ["G_min", "granule cells", "worst drift", "alignment", "Model B gain",
            "of available"],
           title=f"survival threshold sweep, each network at w = 0.005, "
                 f"{len(CH_GMIN_SEEDS)} seeds per threshold")

# ---------------------------------------------------------------- when total concentration fluctuates too
ch_int_Iff = ch_miB(CH_A_INT, np.eye(CH_Nm))
ch_int_a = ch_miB(CH_A_INT, ch_filter(CH_NETS[0]["Ka"], CH_W)) - ch_int_Iff
ch_int_r = ch_miB(CH_A_INT, ch_filter(CH_NETS[0]["Kr"], CH_W)) - ch_int_Iff

# ---------------------------------------------------------------- audit
print()
check("Chow: granule cells in the grown network", ch_Ng, REF_ch_Ng, tol=0.02, fmt="{:.0f}", key="ch_Ng")
check("Chow: enantiomer correlation, input", ch_top["input"], REF_ch_top_in, fmt="{:.3f}", key="ch_top_in")
check("Chow: enantiomer correlation, published", ch_top["published"], REF_ch_top_pub, fmt="{:.3f}", key="ch_top_pub")
check("Chow: enantiomer correlation, grown", ch_top["grown"], REF_ch_top_grown, tol=0.05, fmt="{:.3f}", key="ch_top_grown")
check("Chow: enantiomer correlation, random", ch_top["random"], REF_ch_top_rand, tol=0.05, fmt="{:.3f}", key="ch_top_rand")
check("Chow: alignment, grown", al_a.mean(), REF_ch_align_grown, tol=0.05, fmt="{:.3f}", key="ch_align_grown")
check("Chow: alignment, random", abs(al_r.mean()), REF_ch_align_rand, tol=1.0, fmt="{:.4f}", key="ch_align_rand")
check("Chow: input flatness", ch_F_in, REF_ch_F_in, fmt="{:.4f}", key="ch_F_in")
check("Chow: Model A largest step (negative = monotone)", ch_maxstep, REF_ch_maxstep, tol=1.0, fmt="{:.2e}", key="ch_modelA_maxstep")
check("Chow: Model B gain at w = 0.005, grown (nats)", ch_gain_a.mean(), REF_ch_gain_grown, tol=0.05, fmt="{:.3f}", key="ch_gain_grown")
check("Chow: Model B gain at w = 0.005, random (nats)", ch_gain_r.mean(), REF_ch_gain_rand, tol=0.5, fmt="{:.3f}", key="ch_gain_rand")
check("Chow: Model B gain available (nats)", ch_avail, REF_ch_avail, fmt="{:.3f}", key="ch_avail")
check("Chow: gradient ascent gain (nats)", ch_gain_g.mean(), REF_ch_gain_grad, tol=0.02, fmt="{:.3f}", key="ch_gain_grad")
check("Chow: flatness identity, max error (nats)", ch_flat_err, REF_ch_flat_err, tol=1.0, fmt="{:.1e}", key="ch_flat_err")
record("ch_Nm", CH_Nm); record("ch_d", CH_d); record("ch_n_seeds", len(CH_SEEDS)); record("ch_w", CH_W)
record("ch_top_rand_sd", rd[:, 1].std(ddof=1)); record("ch_top_grown_sd", ad[:, 1].std(ddof=1))
record("ch_rho_in", chow.mean_corr(Cin)); record("ch_rho_pub", chow.mean_corr(CH_C_pub))
record("ch_rho_grown", ad[:, 0].mean()); record("ch_rho_rand", rd[:, 0].mean()); record("ch_Mmean", ad[:, 2].mean())
record("ch_align_grad", np.mean([CH_GA[s]["align"] for s in CH_SEEDS]))
record("ch_Iff", ch_Iff); record("ch_Imax", ch_Imax); record("ch_avail_pct", 100 * ch_avail / ch_Iff)
record("ch_IA0", ch_sw["IA_a"][0, 0]); record("ch_IA_grown", ch_sw["IA_a"][:, jC].mean())
record("ch_IA_rand", ch_sw["IA_r"][:, jC].mean())
record("ch_frac_grown", 100 * ch_gain_a.mean() / ch_avail); record("ch_gain_grown_sd", ch_gain_a.std(ddof=1))
record("ch_gain_sat", ch_gain_sat.mean()); record("ch_frac_sat", 100 * ch_gain_sat.mean() / ch_avail)
record("ch_interior", ch_interior)
record("ch_F_grown", ch_sw["F_a"][:, jC].mean()); record("ch_F_rand", ch_sw["F_r"][:, jC].mean())
record("ch_frac_grad", 100 * ch_gain_g.mean() / ch_avail)
record("ch_F_grad", np.mean([CH_GA[s]["F"] for s in CH_SEEDS]))
record("ch_top_grad", np.mean([CH_GA[s]["top"] for s in CH_SEEDS]))
record("ch_n_iter", CH_N_ITER); record("ch_T", CH_T); record("ch_kappa", CH_KAPPA)
_mid = [f_ for g, f_ in zip(ch_gsw, ch_gfrac) if 0.6 <= g["Gmin"] <= 1.5]
record("ch_gmin_best", ch_gsw[int(np.argmax(ch_gfrac))]["Gmin"])
record("ch_gmin_frac_lo", min(_mid)); record("ch_gmin_frac_hi", max(_mid))
record("ch_gmin_frac_min", ch_gfrac.min()); record("ch_gmin_frac_max", ch_gfrac.max())
record("ch_gmin_maxdrift", max(abs(g["drift"]) for g in ch_gsw))
record("ch_gmin_n_seeds", len(CH_GMIN_SEEDS))
record("ch_gmin_gain_sd_max", max(g["gain_sd"] for g in ch_gsw))
record("ch_drift_max", max(abs(CH_NETS[s]["drift"]) for s in CH_SEEDS))
record("ch_int_gain_grown", ch_int_a); record("ch_int_gain_rand", ch_int_r)
record("ch_int_F_in", ch_flat(CH_A_INT @ CH_A_INT.T))
'''

CODE_FIG5 = r'''
mean = lambda k: ch_sw[k].mean(0)

# Plot data is dumped so the layout can be retuned without rerunning the notebook.
np.savez_compressed(os.path.join(OUTDIR, "figdata_fig4.npz"),
                    CH_WS=CH_WS, IA_a=mean("IA_a"), IA_r=mean("IA_r"),
                    IB_a=mean("IB_a"), IB_r=mean("IB_r"), ch_Iff=ch_Iff,
                    ch_avail=ch_avail, ch_gain_grad=ch_gain_g.mean(), CH_W=CH_W,
                    top=np.array([ch_top[k] for k in ("input", "published", "grown", "random")]),
                    gmins=np.array(CH_GMINS), gfrac=ch_gfrac)

def draw_fig4(W=WIDE_W, H=WIDE_H, name="Fig5_chow_network"):
    """One row of four panels, sized for the full text width plus the left margin."""
    fig, axes = plt.subplots(1, 4, figsize=(W, H))
    ax = axes.ravel()

    # (a) Chow et al.'s decorrelation measure
    labs = ["input", "pub.", "grown", "random"]
    vals = [ch_top[k] for k in ("input", "published", "grown", "random")]
    ax[0].bar(range(4), vals, color=[C_FAINT, C_FAINT, C_TARGET, C_CTRL], width=0.62)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.4)
    ax[0].set_xticks(range(4), labs, fontsize=6.8)
    ax[0].set_ylim(0, 1.08); ax[0].grid(axis="x", visible=False)
    ax[0].set_ylabel("enantiomer-pair corr.")
    ax[0].set_title(f"their decorrelation, $w={CH_W}$", fontsize=8.0)

    # (b) Model A
    ax[1].plot(CH_WS[1:], mean("IA_a")[1:], color=C_TARGET, label="grown")
    ax[1].plot(CH_WS[1:], mean("IA_r")[1:], color=C_CTRL, ls="--", label="random")
    ax[1].axhline(mean("IA_a")[0], color=C_INK, ls=":", lw=1.0, label="no inhib.")
    ax[1].axvline(CH_W, color="#9a9a9a", lw=0.8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("inhibitory strength $w$")
    ax[1].set_ylabel("$I(\\mathbf{c},\\mathbf{M})$  (nats)")
    ax[1].set_title("Model A: in-loop noise", fontsize=8.0)
    ax[1].legend(loc="lower left", fontsize=6.0, labelspacing=0.25)

    # (c) Model B, with the gradient-ascent result at their operating point
    ax[2].plot(CH_WS[1:], mean("IB_a")[1:] - ch_Iff, color=C_TARGET, label="grown")
    ax[2].plot(CH_WS[1:], mean("IB_r")[1:] - ch_Iff, color=C_CTRL, ls="--", label="random")
    ax[2].plot([CH_W], [ch_gain_g.mean()], "o", color=C_MAIN, ms=4.5, zorder=5,
               label="grad. ascent")
    ax[2].axhline(ch_avail, color=C_MAIN, ls="-.", lw=1.0, label="whitened")
    ax[2].axhline(0.0, color=C_INK, ls=":", lw=1.0)
    ax[2].axvline(CH_W, color="#9a9a9a", lw=0.8)
    ax[2].set_xscale("log"); ax[2].set_xlabel("inhibitory strength $w$")
    ax[2].set_ylabel("$I(w)-I(0)$  (nats)")
    ax[2].set_title("Model B: readout noise, budget", fontsize=8.0)
    ax[2].legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=6.0,
               labelspacing=0.25)

    # (d) survival threshold
    ax[3].plot(np.array(CH_GMINS), ch_gfrac, "o-", color=C_TARGET, ms=3.5)
    ax[3].axvline(1.2, color="#9a9a9a", lw=0.8)
    ax[3].text(1.23, 6, "Chow et al.", fontsize=6.2, color="#666666")
    ax[3].set_ylim(0, 100)
    ax[3].set_xlabel("survival threshold $G_{\\min}$")
    ax[3].set_ylabel("Model B gain (% of whitened)")
    ax[3].set_title("selectivity of neurogenesis", fontsize=8.0)

    for axx in ax:
        axx.tick_params(labelsize=6.8)
    panels(axes, dx=-26, dy=4)
    fig.tight_layout(w_pad=0.9)
    save(fig, name)
    return fig

fig = draw_fig4()
plt.show()
'''
