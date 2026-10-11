"""Cell sources, part D: sign obstruction, stability and readout gap, chaining, manifest."""

MD_SIGN = r"""
## 7. A sign obstruction on how flat the output can get

The flatness identity says the benefit is capped by perfect whitening, at
$\mathcal F = 1$. Nothing in it says perfect whitening is reachable. Exact whitening
requires $\mathbf I + w\,\mathbf W^{(gm)\top}\mathbf W^{(gm)} \propto \Sigma_c^{1/2}$, and
the left-hand side is non-negative entrywise, because granule synapses are inhibitory and a
mitral cell can be inhibited through them but never disinhibited. A generic covariance
square root has negative off-diagonal entries, and where it does, no non-negative Gram
matrix can match it at any scale.

Whether this bites therefore depends on the stimulus statistics, and we measure it: the
fraction of off-diagonal entries of $\Sigma_c^{1/2}$ that are negative, for generic
(Wishart) covariances and for the block-structured covariances that odor sources produce,
across a range of population sizes. We use 2000 draws at each of five population sizes.
"""

CODE_SIGN = r'''
def frac_neg_offdiag(Sig):
    ev, U = np.linalg.eigh(Sig)
    S12 = U @ np.diag(np.sqrt(np.maximum(ev, 0))) @ U.T
    off = ~np.eye(len(Sig), dtype=bool)
    return float((S12[off] < 0).mean())

def random_block_cov(N, nb, rng, bg=0.2):
    blk = np.repeat(np.arange(nb), N // nb)
    Sig = bg * np.eye(N)
    for b in range(nb):
        pool = np.where(blk == b)[0]
        Sig[np.ix_(pool, pool)] += 0.6 + 0.3 * rng.random()
    return Sig

N_DRAWS = 2000
sizes = [10, 20, 50, 100, 200]
rng_sign = np.random.default_rng(8)
rows, gen_by_N, blk_by_N = [], {}, {}
t0 = time.time()
for N in sizes:
    nb = max(2, N // 10)
    g = [frac_neg_offdiag(np.cov(rng_sign.standard_normal((N, 3 * N)))) for _ in range(N_DRAWS)]
    b = [frac_neg_offdiag(random_block_cov(N, nb, rng_sign)) for _ in range(N_DRAWS)]
    gen_by_N[N], blk_by_N[N] = np.array(g), np.array(b)
    rows.append([N, f"{100*np.mean(g):.1f} +/- {100*np.std(g,ddof=1):.1f}",
                 f"{100*np.mean(b):.1f} +/- {100*np.std(b,ddof=1):.1f}"])
sign_secs = time.time() - t0

frac_paper_sigma = frac_neg_offdiag(Sigma_c)
show_table(rows, ["N", "generic Wishart (% negative)", "block-structured (% negative)"],
           title=f"negative off-diagonals of the covariance square root "
                 f"({N_DRAWS} draws per size, {sign_secs:.0f} s)")
print()
check("generic Wishart, N=50, negative off-diagonals (%)",
      100 * gen_by_N[50].mean(), REF_sign_gen, tol=0.1, fmt="{:.1f}", key="sign_generic")
check("block-structured, N=50, negative off-diagonals (%)",
      100 * blk_by_N[50].mean(), REF_sign_blk, tol=1.0, fmt="{:.2f}", key="sign_block")
check("this paper's Sigma_c, negative off-diagonals (%)",
      100 * frac_paper_sigma, REF_sign_paper, tol=1.0, fmt="{:.2f}", key="sign_paper")
record("sign_generic_sd", 100 * gen_by_N[50].std(ddof=1))
for N in sizes:
    record(f"sign_generic_N{N}", 100 * gen_by_N[N].mean())
    record(f"sign_block_N{N}", 100 * blk_by_N[N].mean())
record("N_DRAWS", N_DRAWS); record("n_sizes", len(sizes))
'''

MD_STAB = r"""
## 8. Reciprocity and the stability of the two-population loop

Everything above describes the mitral population after the granule cells have been
eliminated adiabatically. That reduction is only licensed if the full two-population loop is
stable, and reciprocity settles the question completely. Rescaling the granule activity by
$\sqrt w$ makes the coupling skew-symmetric, so the full spectrum is
$\{-1 \pm i\sqrt w\,\mu_k\}\cup\{-1\}$ with $\mu_k$ the singular values of
$\mathbf W^{(gm)}$: every eigenvalue has real part exactly $-1$, for any connectivity and
any $w$. We verify the predicted spectrum numerically, and check that the reduced closed form
reproduces the full-state information on the same network.

"""

CODE_STAB = r'''
N_m_rg, N_g_rg, sigma2_rg, NC_rg, N_SEEDS_RG = 30, 90, 0.09, 4, 30
blocks_rg = np.repeat(np.arange(5), N_m_rg // 5)
Sigma_rg = generate_odor_covariance(N_m_rg, 5)

def stacked(W_gm, w):
    N_mm, N_gg = W_gm.shape[1], W_gm.shape[0]
    return np.block([[-np.eye(N_mm), -w * W_gm.T],
                     [W_gm,          -np.eye(N_gg)]])

def readout_gap(W_gm, w, convention="symmetrised"):
    N_mm, N_gg = W_gm.shape[1], W_gm.shape[0]
    W_full = stacked(W_gm, w)
    K_inj = np.vstack([np.eye(N_mm), np.zeros((N_gg, N_mm))])
    B = -np.linalg.solve(W_full, K_inj)
    if convention == "symmetrised":
        Seta = sigma2_rg * np.diag(np.concatenate([np.ones(N_mm),
                                                   (1.0 / w) * np.ones(N_gg)]))
    else:
        Seta = sigma2_rg * np.eye(N_mm + N_gg)
    Src = solve_continuous_lyapunov(W_full, -Seta)
    SR = Src + B @ Sigma_rg @ B.T
    mm = slice(0, N_mm)
    I_R = 0.5 * (np.linalg.slogdet(SR)[1] - np.linalg.slogdet(Src)[1])
    I_M = 0.5 * (np.linalg.slogdet(SR[mm, mm])[1] - np.linalg.slogdet(Src[mm, mm])[1])
    return I_R, I_M

# ---------------------------------------------------------------- spectrum
w_spec = [0.005, 0.05, 0.2, 1.0]
W_rg0 = make_wiring(N_m_rg, N_g_rg, NC_rg, "random", seed=3)
mu = np.linalg.svd(W_rg0, compute_uv=False)
spec_err, spectra = [], {}
for w_ in w_spec:
    ev = np.linalg.eigvals(stacked(W_rg0, w_))
    spectra[w_] = ev
    spec_err.append(np.max(np.abs(ev.real + 1.0)))
    pred_imag = np.sort(np.concatenate([np.sqrt(w_) * mu, -np.sqrt(w_) * mu,
                                        np.zeros(N_m_rg + N_g_rg - 2 * len(mu))]))
    got_imag = np.sort(ev.imag)
    spec_err.append(np.max(np.abs(pred_imag - got_imag)))
spec_err = float(np.max(spec_err))

# ---------------------------------------------------------------- consistency check
# The reduced closed form must reproduce the full-state value on the same network under the
# symmetrized noise convention. This is what licenses the adiabatic reduction numerically.
I_red = 0.5 * np.linalg.slogdet(
    np.eye(N_m_rg) + (2.0 / sigma2_rg)
    * np.linalg.solve(np.eye(N_m_rg) + 0.05 * (W_rg0.T @ W_rg0), Sigma_rg))[1]
IR05, IM05 = readout_gap(W_rg0, 0.05, "symmetrised")

print(f"stability spectrum checked at w in {w_spec}, "
      f"N_m = {N_m_rg}, N_g = {N_g_rg}")
print()
check("max |Re(eigenvalue) + 1| and imag-part error", spec_err, REF_spec_err,
      tol=1.0, fmt="{:.2e}", key="spectrum_error")
check("reduced closed form vs full state, w=0.05", abs(I_red - IR05), REF_red_err,
      tol=1.0, fmt="{:.2e}", key="reduced_vs_full")
record("N_m_rg", N_m_rg); record("N_g_rg", N_g_rg)
record("n_w_spec", len(w_spec))
'''

CODE_MANIFEST = r'''
with open("results.json", "w") as f:
    json.dump(RESULTS, f, indent=1, sort_keys=True)

expected = ["Fig1_framework", "Fig2_two_noise_models", "Fig3_flatness",
            "Fig4_gradient_ascent", "Fig5_chow_network"]
print("figure manifest")
print("=" * 52)
for name in expected:
    p = os.path.join(OUTDIR, name + ".pdf")
    print(f"  {'OK ' if os.path.exists(p) else 'MISSING'}  {p}"
          f"{'' if not os.path.exists(p) else f'   {os.path.getsize(p)/1024:.0f} kB'}")
print()
n_compared = sum(1 for c in CHECKS if c[3] is not None)
record("n_checks", n_compared)
record("n_recorded", len(RESULTS))
with open("results.json", "w") as f:
    json.dump(RESULTS, f, indent=1, sort_keys=True)
print(f"{len(RESULTS)} quantities recorded -> results.json")
print(f"{n_compared} of them compared against the value printed in the manuscript")
if FAILURES:
    print("\nFAILED CHECKS:")
    for f_ in FAILURES:
        print("  " + f_)
    raise AssertionError(f"{len(FAILURES)} check(s) disagree with the manuscript")
print("\nAll checks agree with the values printed in the manuscript.")
print("\nEvery number quoted in the manuscript, including its appendix, appears")
print("above beside its computed value.")
'''
