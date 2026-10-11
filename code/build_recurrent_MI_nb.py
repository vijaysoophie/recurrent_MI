"""Build recurrent_MI_simulations.ipynb.

Pass 1 builds the notebook with no reference values (every check prints "no reference
yet"). Running it writes results.json. Pass 2, invoked with --ref results.json, rebuilds
the notebook with those values hardcoded as the manuscript references, so the shipped
artifact checks every computed number against the number printed in the paper.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nb_parts_a as A
import nb_parts_b as B
import nb_parts_c as C
import nb_parts_d as D
import nb_parts_f as F

# REF_<name> placeholder  ->  key in results.json
REF_MAP = {
    "REF_r_in_same": "r_in_same",
    "REF_I_ff": "I_ff",
    "REF_F_in": "F_in",
    "REF_rel_cond": "mc_rel_cond",
    "REF_rel_marg": "mc_rel_marg",
    "REF_I_theory": "mc_I_theory",
    "REF_I_mc": "mc_I_mc",
    "REF_I_gap": "mc_I_gap",
    "REF_miA0": "miA_w0",
    "REF_miA01": "miA_w01",
    "REF_miAend": "miA_wend",
    "REF_frac_mono": "frac_monotone",
    "REF_worst_inc": "worst_increase",
    "REF_wstar": "w_star",
    "REF_gain_match": "gain_match",
    "REF_gain_rand": "gain_rand",
    "REF_ss_in": "ss_in",
    "REF_ss_rand": "ss_out_random",
    "REF_ss_match": "ss_out_matched",
    "REF_flat_maxerr": "flat_max_err",
    "REF_flat_mederr": "flat_med_err",
    "REF_flat_errstar": "flat_err_wstar",
    "REF_flat_r": "flat_r",
    "REF_flat_r_useful": "flat_r_useful",
    "REF_flat_globalmax": "flat_global_max_err",
    "REF_pearson": "pearson_within",
    "REF_F_out": "F_out",
    "REF_F_in2": "F_in_again",
    "REF_flat_err_sig": "flat_err_sigma01",
    "REF_grad_gain": "grad_gain",
    "REF_grad_gap": "grad_gap_closed",
    "REF_grad_pur0": "grad_purity_start",
    "REF_grad_pur1": "grad_purity_end",
    "REF_chance_pur": "chance_purity",
    "REF_grad_sd": "grad_restart_sd",
    "REF_grad_F": "grad_F",
    "REF_ch_Ng": "ch_Ng",
    "REF_ch_top_in": "ch_top_in",
    "REF_ch_top_pub": "ch_top_pub",
    "REF_ch_top_grown": "ch_top_grown",
    "REF_ch_top_rand": "ch_top_rand",
    "REF_ch_align_grown": "ch_align_grown",
    "REF_ch_align_rand": "ch_align_rand",
    "REF_ch_F_in": "ch_F_in",
    "REF_ch_maxstep": "ch_modelA_maxstep",
    "REF_ch_gain_grown": "ch_gain_grown",
    "REF_ch_gain_rand": "ch_gain_rand",
    "REF_ch_avail": "ch_avail",
    "REF_ch_gain_grad": "ch_gain_grad",
    "REF_ch_flat_err": "ch_flat_err",
}


def substitute(src, ref):
    for placeholder, key in sorted(REF_MAP.items(), key=lambda kv: -len(kv[0])):
        if ref and key in ref:
            val = repr(float(ref[key]))
        else:
            val = "float('nan')"
        src = src.replace(placeholder, val)
    return src


_N = [0]


def _next_id():
    _N[0] += 1
    return f"cell{_N[0]:03d}"


def md(text):
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {},
            "source": text.strip("\n").splitlines(True)}


def code(text, ref):
    return {"cell_type": "code", "id": _next_id(), "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": substitute(text, ref).strip("\n").splitlines(True)}


def build(ref):
    _N[0] = 0
    cells = [
        md(A.MD_TITLE),
        md("## 0. Dependencies and conventions"),
        code(A.CODE_DEPS, ref),
        md("### Shared figure style and the audit machinery\n\n"
           "Cosmetic settings plus two helpers used throughout: `check`, which prints every "
           "computed quantity beside the value quoted in the paper, and `record`, which "
           "collects them all into `results.json` at the end."),
        code(A.CODE_STYLE, ref),
        md(A.MD_SETUP),
        code(A.CODE_SETUP, ref),
        md(A.MD_MC),
        code(A.CODE_MC, ref),
        md(A.MD_FIG1),
        code(B.CODE_FIG1, ref),
        md(B.MD_SWEEP),
        code(B.CODE_SWEEP, ref),
        md("### Pattern decorrelation at a fixed inhibitory strength"),
        code(B.CODE_DECORR, ref),
        md("### Fig 2. Two noise models, opposite verdicts"),
        code(B.CODE_FIG2, ref),
        md(C.MD_FLAT),
        code(C.CODE_FLAT, ref),
        md("### Fig 3. The benefit of inhibition is a spectral-flatness ratio"),
        code(C.CODE_FIG3, ref),
        md("### Robustness of the interior optimum across stimulus ensembles"),
        code(C.CODE_ROBUST, ref),
        md(C.MD_GRAD),
        code(C.CODE_GRAD, ref),
        md("### Fig 4. Gradient ascent finds the source-matched architecture"),
        code(C.CODE_FIG4, ref),
        md(F.MD_CHOW),
        code(F.CODE_CHOW, ref),
        md("### Fig 5. The analysis on Chow et al.'s network"),
        code(F.CODE_FIG5, ref),
        md("## 10. Manifest and audit dump"),
        code(D.CODE_MANIFEST, ref),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default="recurrent_MI_simulations.ipynb")
    a = ap.parse_args()
    ref = json.load(open(a.ref)) if a.ref and os.path.exists(a.ref) else None
    nb = build(ref)
    with open(a.out, "w") as f:
        json.dump(nb, f, indent=1)
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"wrote {a.out}: {len(nb['cells'])} cells ({n_code} code), "
          f"references {'loaded from ' + a.ref if ref else 'not yet available'}")
