# recurrent_MI

Mutual information in linear recurrent circuits. Simulation code and data for:

> Vijay Singh and Martin Tchernookov. *Mutual information in linear recurrent circuits:
> inhibition in the olfactory bulb increases information only against a fixed noise floor.*

Everything the paper reports is produced by one notebook, `recurrent_MI_simulations.ipynb`, which
runs top to bottom with no arguments and is fully seeded.

## What the paper asks

Recurrent inhibition in the olfactory bulb decorrelates glomerular responses. Whether that
decorrelation helps the bulb transmit information turns out to depend on where the noise
enters. We derive a closed-form expression for the mutual information between a stimulus and
the steady state of a linear recurrent circuit, and use it to compare two noise models on the
same network:

- **Model A, noise inside the recurrent loop.** Information falls monotonically with the
  inhibitory gain, for any connectivity. Inhibition can only cost.
- **Model B, readout noise downstream of the circuit plus a fixed activity budget.**
  Information rises to an interior optimum, but only when the granule-cell wiring is matched
  to the structure of the odor sources.

The benefit, where it exists, is exactly the log ratio of the output to the input spectral
flatness. The same formula reproduces three published models of the early olfactory pathway
as limiting cases.

## Quick start

```bash
git clone https://github.com/<user>/recurrent_MI.git
cd recurrent_MI
pip install -r requirements.txt
jupyter lab recurrent_MI_simulations.ipynb      # or: jupyter nbconvert --execute ...
```

Run it from the repository root: the notebook reads `data/`, imports `chow_model.py`, and
writes `figures/` and `results.json` relative to the working directory.

A full run takes **about 50 minutes** on two cores. The expensive parts are the Chow et al.
network (10 seeds grown to steady state, ~24 min), the gradient ascent (100 restarts of 8000
iterations, ~14 min) and a Monte Carlo validation of the closed form (~5 min). Only numpy,
scipy and matplotlib are needed to run the science; the remaining requirements are for
regenerating the notebook from its sources.

## The notebook checks itself

Every quantity the paper quotes is computed here, recorded under a key, and written to
`results.json`. Fifty of them are additionally compared against the value printed in the
paper, which is hardcoded in the notebook, and the final cell raises if any comparison fails:

```
  [OK  ] Model B matched, w*                             computed      0.0058   manuscript      0.0058
  [OK  ] Chow: enantiomer correlation, input             computed       0.798   manuscript       0.798
  ...
  All checks agree with the values printed in the manuscript.
```

So a reader who reruns this learns immediately whether the paper's numbers still hold, rather
than having to audit them by hand. The published run records 182 quantities and passes 50 of
50 checks.

The notebook itself is generated from plain Python modules in `code/`, in two passes: the
first produces `results.json`, the second rebuilds the notebook with those values hardcoded
as the references it checks against:

```bash
python3 code/build_recurrent_MI_nb.py                        # pass 1, no reference values
# run the notebook, which writes results.json
python3 code/build_recurrent_MI_nb.py --ref results.json     # pass 2, references hardcoded
```

Run both from the repository root, since the notebook is written to the working directory.
Editing a cell means editing the corresponding `code/nb_parts_*.py` and regenerating, not
editing the `.ipynb`.

## Layout

| path | what it is |
| --- | --- |
| `recurrent_MI_simulations.ipynb` | the notebook, with the outputs of the published run |
| `chow_model.py` | our implementation of the neurogenesis rule of Chow et al. (2012) |
| `code/nb_parts_*.py` | the notebook's cell sources, which `build_recurrent_MI_nb.py` assembles |
| `code/build_recurrent_MI_nb.py` | assembles the notebook; `--ref results.json` for the second pass |
| `code/preview_rows.py` | redraws Figs 3 and 4 from the dumped plot data, for layout work |
| `data/chow2012_stimuli.npz` | the eight glomerular activation patterns, see below |
| `figures/` | the figures of the published run, plus `figdata_*.npz` plot dumps |
| `results.json` | every recorded quantity from the published run |

## Data

The one real dataset is `data/chow2012_stimuli.npz`: eight glomerular activation patterns
across 424 glomeruli, the stimuli that drive all of the simulations in Chow, Wick and Riecke
(2012), *Neurogenesis drives stimulus decorrelation in a model of the olfactory bulb*, PLOS
Computational Biology 8(3):e1002398.

**These patterns were recovered by us from the published figures of that paper, not obtained
from its authors.** The original 2-deoxyglucose maps are not distributed with it, so we
decoded the glomerular maps from the figures and validated the result against the one
published statistic of the input ensemble, the correlation matrix of their Fig 2Ci. Our
recovered patterns reproduce it to 0.015 in the largest absolute difference over the 28 odor
pairs, within one step of that figure's color scale. The procedure is described in the
Methods of our paper. Chow et al. is published under CC BY, which is what permits
redistribution of material derived from its figures.

The archive also carries the recovered published correlation matrices, the down-sampled
50-glomerulus version used for the smaller network, and the index arrays relating them. Treat
the patterns as our reconstruction, with the uncertainty that implies, rather than as the
original data.

## Reusing the formulas

The two information formulas and the flatness measure are small and self-contained, defined
in the notebook's setup cell:

```python
mi_inloop(Sigma_c, W_eff, sigma)       # Model A: noise inside the loop
mi_budget(Sigma_c, W_eff, sigma, P)    # Model B: readout noise, fixed activity budget
spectral_flatness(Sigma)               # geometric over arithmetic mean of the eigenvalues
```

`W_eff = I + w W.T @ W` for reciprocal granule-mitral connectivity `W`. Note that Model A as
written assumes `W_eff` is symmetric, which reciprocity gives; for a general stable `W_eff`
the stationary noise covariance has to come from the Lyapunov equation instead.

## Citing

Please cite the paper. If you use the recovered stimulus patterns, cite Chow et al. (2012) as
well as this repository, since the patterns are derived from their figures.

## License

MIT, see `LICENSE`. The recovered stimulus data derives from a CC BY publication; attribution
to Chow et al. (2012) is required when redistributing it.
