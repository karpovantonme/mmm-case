# Marketing Mix Modeling: does the budget split make sense?

A Bayesian media-mix model taken past "the model fits" to an actual budget decision, with the honesty checks a real client engagement would need.

## The two numbers

**Reallocating this quarter's budget buys +0.34%.** At the same $62,472 quarterly spend, the model-optimal split beats the current historical split by a third of a percent. That is close to nothing, and the reason is worth more than the number: the current split already sits near the point where the two channels' marginal returns balance.

**The next dollar is worth 1.5x more in one channel than the other.** At today's spend levels, an incremental dollar into x1 returns $0.52 of response against $0.34 for x2.

These answer different questions. Shuffling existing money is not worth relitigating a signed media contract. Deciding where a budget *increase* goes is a live call, and this is the number for it.

![Marginal ROI](assets/marginal_roi.png)

## The check most MMM write-ups skip

Before trusting a model on data with no answer key, run it on data where the answer is known. I generated 156 weeks of spend and response with fixed adstock rates, saturation curves and channel contributions, then asked the model to recover them.

| channel | true total | recovered | error | true value inside 94% HDI |
|---|---|---|---|---|
| x1 | 389,754 | 413,295 | +6.0% | yes |
| x2 | 230,715 | 164,266 | -28.8% | **no**, about 4% outside the upper bound |

![Recovery check](assets/recovery_check.png)

Both fits converged cleanly, r_hat 1.0 and zero divergences, so the x2 gap is not a sampling failure. It is identification: x2 carries a stronger carryover (adstock 0.6 against x1's 0.4), and 156 weekly observations do not pin down its saturation curve from spend alone. That is a documented MMM limitation, and in practice x2 is exactly the channel you run a lift test on.

Reported rather than tuned away, because a suspiciously clean result is less useful than an honest one.

## Method

| | |
|---|---|
| Model | `pymc-marketing` 0.19.4, `multidimensional.MMM` (the current class, `mmm.MMM` is deprecated) |
| Mechanics | adstock (spend keeps working for weeks after it lands) and saturation (every channel has a ceiling) |
| Diagnostics | r_hat, ESS, divergences, plus the recovery check above |
| Optimization | `BudgetOptimizer`, bounded to the spend range each channel has actually seen |

Two data tracks on purpose. **Track A** is `pymc-marketing`'s bundled demo set, 179 weeks and 2 channels, as an end-to-end sanity check. **Track B** is the synthetic data with known ground truth described above. Neither is real client data and neither is presented as one.

Budget setup was chosen to avoid the most common MMM mistake, extrapolating past the range the saturation curve was calibrated on: horizon of 13 weeks, total equal to 13x the historical average weekly spend rather than a hypothetical larger number, and a per-channel cap at 13x that channel's historical *maximum* weekly spend. The optimizer is never asked to price a spend level it has never seen.

![Waterfall decomposition](assets/waterfall_decomposition.png)
![Budget reallocation](assets/budget_reallocation.png)

## What changes with real client data

- Recovery on synthetic data is the calibration step, not the deliverable. Real data never comes with an answer key, so this runs first and once.
- Whichever channel comes back weakly identified gets a lift test, not a point estimate anyone acts on.
- Budget optimization re-runs as spend history grows. The safe bound grows with it, and so does confidence in the recommendation.
- Both headline numbers get checked against how flexible the media plan actually is. A 0.34% reallocation gain is not worth reopening a contract. A 1.5x marginal gap is worth acting on for net-new budget.

## Limitations

Track B is synthetic. It shows the method recovers a known truth. It shows nothing about these two channels in the real world.

The optimizer never proposes spend outside a channel's historical range, by design. No output here should be read as "spend more per week than we ever have".

x2's saturation curve carries real uncertainty. Any decision resting on its absolute ceiling needs a lift test alongside this model.

## Why MMM at all

Click-based attribution stopped being trustworthy after iOS 14.5, cookie deprecation and walled gardens. MMM answers the budget question from spend and outcome data alone, with no tracking pixels involved.

## Repo

| | |
|---|---|
| `notebooks/mmm_case.ipynb` | the executed notebook: data, model, diagnostics, recovery, visuals, optimization |
| `scripts/build_notebook.py` | generates the notebook from source, keeps cell sources under version control cleanly |
| `data/` | bundled demo set, generated synthetic set, and its ground truth |
| `requirements.txt` | pinned dependencies. Exact dev environment in `pyproject.toml` and `uv.lock`, built with [uv](https://github.com/astral-sh/uv) |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/mmm_case.ipynb
```

MIT.
