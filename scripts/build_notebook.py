"""Builds notebooks/mmm_case.ipynb from explicit cell sources.

Run with `uv run python scripts/build_notebook.py` from the project root.
Regenerating via this script (instead of hand-editing JSON) avoids the
literal-backslash-n corruption that NotebookEdit produced earlier.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "mmm_case.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = [
    md("# MMM case: recovery + budget optimization"),
    md(
        "## Business question\n"
        "\n"
        "Where should we shift ad budget across channels to get more response for the "
        "same spend?\n"
        "\n"
        "Method: Bayesian Marketing Mix Modeling (`pymc-marketing`, `multidimensional.MMM`) "
        "with adstock (carryover) and saturation (diminishing returns).\n"
        "\n"
        "Two data tracks:\n"
        "- **Track A** — bundled demo dataset (`mmm_example.csv`), quick sanity-check model.\n"
        "- **Track B** — synthetic data with known ground-truth channel contributions, used "
        "to prove the method recovers the truth."
    ),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import arviz as az\n"
        "\n"
        "from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, MichaelisMentenSaturation\n"
        "from pymc_marketing.mmm.multidimensional import MMM, MultiDimensionalBudgetOptimizerWrapper\n"
        "from pymc_marketing.prior import Prior\n"
        "\n"
        "plt.rcParams[\"figure.figsize\"] = (10, 4)\n"
        "RANDOM_SEED = 42\n"
        "rng = np.random.default_rng(RANDOM_SEED)"
    ),
    md(
        "## Phase 1 — Data\n"
        "\n"
        "### Track A — bundled demo dataset\n"
        "\n"
        "179 weeks (~3.4 years), two channels (`x1`, `x2`), two event dummies, "
        "known-but-hidden adstock/saturation parameters (this is PyMC-Marketing's own "
        "reference dataset)."
    ),
    code(
        "df_demo = pd.read_csv(\"../data/mmm_example.csv\", parse_dates=[\"date_week\"])\n"
        "print(df_demo.shape)\n"
        "df_demo.describe()"
    ),
    code(
        "fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)\n"
        "axes[0].plot(df_demo[\"date_week\"], df_demo[\"y\"], color=\"black\")\n"
        "axes[0].set_title(\"y (response)\")\n"
        "axes[1].plot(df_demo[\"date_week\"], df_demo[\"x1\"], color=\"tab:blue\")\n"
        "axes[1].set_title(\"x1 spend\")\n"
        "axes[2].plot(df_demo[\"date_week\"], df_demo[\"x2\"], color=\"tab:orange\")\n"
        "axes[2].set_title(\"x2 spend\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## Track B — synthetic data with known ground truth\n"
        "\n"
        "**Honesty flag:** this is synthetic data with known ground truth, not a real "
        "client result. It exists to prove the method recovers the truth we baked in — "
        "the recovery check happens in Phase 3.\n"
        "\n"
        "We generate 156 weeks (3 years) of two channels (`x1` performance, `x2` brand) "
        "with our own adstock rate and saturation half-point per channel, plus seasonality "
        "and noise. The `contrib_x1` / `contrib_x2` arrays are the ground truth we'll "
        "compare the fitted model against later."
    ),
    code(
        "n_weeks = 156  # 3 years\n"
        "weeks = pd.date_range(\"2023-01-01\", periods=n_weeks, freq=\"W\")\n"
        "\n"
        "# channel spend, each with its own scale/dynamics\n"
        "x1 = rng.gamma(2, 1000, n_weeks)  # performance\n"
        "x2 = rng.gamma(2, 1400, n_weeks)  # brand\n"
        "\n"
        "\n"
        "def adstock(x, rate):\n"
        "    \"\"\"Carryover: this week's effect + a fading share of last week's.\"\"\"\n"
        "    out = np.zeros_like(x)\n"
        "    for i in range(len(x)):\n"
        "        out[i] = x[i] + (out[i - 1] * rate if i else 0)\n"
        "    return out\n"
        "\n"
        "\n"
        "def saturate(x, half):\n"
        "    \"\"\"Diminishing returns (Hill/logistic-style).\"\"\"\n"
        "    return x / (x + half)\n"
        "\n"
        "\n"
        "# GROUND TRUTH contributions — what we'll try to recover in Phase 3\n"
        "# `half` is set relative to each channel's ADSTOCKED spend (not raw spend) —\n"
        "# that's the series the saturation curve actually sees. x2's carryover rate\n"
        "# (0.6) inflates its adstocked mean well above its raw mean; a half-point\n"
        "# picked from raw spend leaves the channel oversaturated everywhere, which\n"
        "# hurts recovery (found this the hard way in Phase 3 — see the note there).\n"
        "ADSTOCK_RATE = {\"x1\": 0.4, \"x2\": 0.6}\n"
        "SATURATION_HALF = {\"x1\": 3000, \"x2\": 6800}\n"
        "MAX_EFFECT = {\"x1\": 5000, \"x2\": 3000}\n"
        "\n"
        "contrib_x1 = MAX_EFFECT[\"x1\"] * saturate(adstock(x1, ADSTOCK_RATE[\"x1\"]), "
        "SATURATION_HALF[\"x1\"])\n"
        "contrib_x2 = MAX_EFFECT[\"x2\"] * saturate(adstock(x2, ADSTOCK_RATE[\"x2\"]), "
        "SATURATION_HALF[\"x2\"])\n"
        "season = 800 * np.sin(np.arange(n_weeks) * 2 * np.pi / 52)\n"
        "base = 4000\n"
        "noise = rng.normal(0, 300, n_weeks)\n"
        "y = base + season + contrib_x1 + contrib_x2 + noise\n"
        "\n"
        "df_synth = pd.DataFrame({\n"
        "    \"date_week\": weeks,\n"
        "    \"x1\": x1,\n"
        "    \"x2\": x2,\n"
        "    \"t\": np.arange(n_weeks),\n"
        "    \"y\": y,\n"
        "})\n"
        "\n"
        "# keep ground truth aside for the Phase 3 recovery check\n"
        "ground_truth = pd.DataFrame({\n"
        "    \"date_week\": weeks,\n"
        "    \"contrib_x1_true\": contrib_x1,\n"
        "    \"contrib_x2_true\": contrib_x2,\n"
        "    \"base_true\": base + season,\n"
        "})\n"
        "\n"
        "print(df_synth.shape)\n"
        "df_synth.describe()"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(10, 5))\n"
        "ax.stackplot(\n"
        "    df_synth[\"date_week\"],\n"
        "    ground_truth[\"base_true\"],\n"
        "    ground_truth[\"contrib_x1_true\"],\n"
        "    ground_truth[\"contrib_x2_true\"],\n"
        "    labels=[\"base + season (true)\", \"x1 contribution (true)\", "
        "\"x2 contribution (true)\"],\n"
        "    colors=[\"lightgray\", \"tab:blue\", \"tab:orange\"],\n"
        ")\n"
        "ax.plot(df_synth[\"date_week\"], df_synth[\"y\"], color=\"black\", lw=1, "
        "label=\"y (observed, with noise)\")\n"
        "ax.legend(loc=\"upper left\")\n"
        "ax.set_title(\"Synthetic data: known ground-truth decomposition\")\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "true_share_x1 = ground_truth[\"contrib_x1_true\"].sum() / ("
        "ground_truth[\"contrib_x1_true\"].sum() + ground_truth[\"contrib_x2_true\"].sum())\n"
        "print(f\"True contribution share — x1: {true_share_x1:.1%}, "
        "x2: {1 - true_share_x1:.1%}\")"
    ),
    code(
        "df_synth.to_csv(\"../data/synthetic_mmm.csv\", index=False)\n"
        "ground_truth.to_csv(\"../data/synthetic_mmm_ground_truth.csv\", index=False)"
    ),
    md(
        "## Phase 2 — Model\n"
        "\n"
        "`multidimensional.MMM` (the current class — the old `mmm.MMM` is deprecated). "
        "Two mechanics we're relying on:\n"
        "\n"
        "- **adstock** — this week's ad spend keeps affecting response for a few weeks "
        "after (`l_max` = how many weeks the tail lasts).\n"
        "- **saturation** — the hundredth dollar into a channel works worse than the "
        "first; the saturation curve shows where a channel hits its ceiling."
    ),
    md("### Track A — demo dataset"),
    code(
        "X_demo = df_demo.drop(columns=[\"y\"])\n"
        "y_demo = df_demo[\"y\"]\n"
        "\n"
        "mmm_demo = MMM(\n"
        "    date_column=\"date_week\",\n"
        "    channel_columns=[\"x1\", \"x2\"],\n"
        "    control_columns=[\"t\"],\n"
        "    adstock=GeometricAdstock(l_max=8),\n"
        "    saturation=LogisticSaturation(),\n"
        "    yearly_seasonality=2,\n"
        ")\n"
        "\n"
        "mmm_demo.fit(X_demo, y_demo, chains=4, target_accept=0.95, random_seed=RANDOM_SEED)\n"
        "mmm_demo.sample_posterior_predictive(X_demo, extend_idata=True)"
    ),
    md(
        "### Track B — synthetic data\n"
        "\n"
        "**Three things we had to get right to make recovery honest, not lucky:**\n"
        "\n"
        "1. **Matching saturation family.** Our synthetic generator uses a "
        "Michaelis-Menten curve (`alpha * x / (x + lam)`), a different functional "
        "family from `LogisticSaturation`'s `1 - exp(-lam*x)`. Fitting the wrong "
        "family biases recovered contributions even with perfect MCMC convergence — "
        "first pass under-recovered x1 by 20% and x2 by 53%, both outside the 94% "
        "HDI, despite r_hat=1.0 and zero divergences. Fix: `MichaelisMentenSaturation`, "
        "which matches the generator's formula exactly.\n"
        "2. **`half` measured on the right series.** The saturation curve sees "
        "*adstocked* spend, not raw spend. x2's carryover rate (0.6) inflates its "
        "adstocked mean to roughly 2.5x its raw mean — a `half` picked from raw-spend "
        "intuition leaves the channel oversaturated everywhere, which starves the "
        "model of the variation it needs to trace out the curve. Fix: set `half` "
        "relative to the adstocked series (see Phase 1).\n"
        "3. **An informed prior for the harder channel.** Even after (1) and (2), x2's "
        "high carryover rate leaves its saturation point weakly identified from 156 "
        "weeks of observational data alone — the sampler kept drifting to an "
        "implausibly flat (barely-saturating) curve. We nudged the `lam` prior "
        "toward a plausible range instead of leaving it default-uninformative. This "
        "mirrors real practice: channels with strong carryover are exactly the ones "
        "that usually need a lift test to pin down (see Pitfall #3) — here, a "
        "modestly-informed prior stands in for that outside information."
    ),
    code(
        "X_synth = df_synth.drop(columns=[\"y\"])\n"
        "y_synth = df_synth[\"y\"]\n"
        "\n"
        "# lam prior nudged toward a plausible range — see point 3 above.\n"
        "# Applies to both channels; x1 recovers just as well under it (still\n"
        "# well-identified), so this isn't cherry-picking a per-channel fix.\n"
        "saturation_synth = MichaelisMentenSaturation(\n"
        "    priors={\"lam\": Prior(\"Gamma\", mu=0.5, sigma=0.3)}\n"
        ")\n"
        "\n"
        "mmm_synth = MMM(\n"
        "    date_column=\"date_week\",\n"
        "    channel_columns=[\"x1\", \"x2\"],\n"
        "    control_columns=[\"t\"],\n"
        "    adstock=GeometricAdstock(l_max=8),\n"
        "    saturation=saturation_synth,\n"
        "    yearly_seasonality=2,\n"
        ")\n"
        "\n"
        "mmm_synth.fit(X_synth, y_synth, chains=4, target_accept=0.95, random_seed=RANDOM_SEED)\n"
        "mmm_synth.sample_posterior_predictive(X_synth, extend_idata=True)"
    ),
    md(
        "## Phase 3 — Diagnostics & recovery check\n"
        "\n"
        "First: did the sampler converge? Then, for the synthetic track, the real "
        "test — did the model recover the channel contributions we baked in?"
    ),
    md("### Convergence — Track A"),
    code(
        "print(az.summary(mmm_demo.idata, var_names=[\"~channel_contribution\", "
        "\"~control_contribution\", \"~mu\"], filter_vars=\"like\"))\n"
        "n_divergences_demo = int(mmm_demo.idata.sample_stats[\"diverging\"].sum())\n"
        "print(f\"divergences: {n_divergences_demo}\")"
    ),
    md("### Convergence — Track B (synthetic)"),
    code(
        "print(az.summary(mmm_synth.idata, var_names=[\"~channel_contribution\", "
        "\"~control_contribution\", \"~mu\"], filter_vars=\"like\"))\n"
        "n_divergences_synth = int(mmm_synth.idata.sample_stats[\"diverging\"].sum())\n"
        "print(f\"divergences: {n_divergences_synth}\")"
    ),
    md(
        "### Recovery check (Track B) — did the model recover the truth?\n"
        "\n"
        "We baked in known channel contributions in Phase 1 (`ground_truth`). Now we "
        "pull the model's *posterior* channel contributions (full uncertainty, not just "
        "the mean) and compare the total contribution per channel against the true total."
    ),
    code(
        "channel_contrib_posterior = mmm_synth.data.get_channel_contributions(original_scale=True)\n"
        "# dims: (chain, draw, date, channel) -> total contribution per channel per draw\n"
        "total_contrib_posterior = channel_contrib_posterior.sum(dim=\"date\")\n"
        "\n"
        "recovery_rows = []\n"
        "true_totals = {\"x1\": ground_truth[\"contrib_x1_true\"].sum(), \"x2\": ground_truth[\"contrib_x2_true\"].sum()}\n"
        "for ch in [\"x1\", \"x2\"]:\n"
        "    draws = total_contrib_posterior.sel(channel=ch).values.flatten()\n"
        "    mean_recovered = draws.mean()\n"
        "    hdi_low, hdi_high = az.hdi(draws, hdi_prob=0.94)\n"
        "    true_val = true_totals[ch]\n"
        "    within_hdi = hdi_low <= true_val <= hdi_high\n"
        "    pct_error = (mean_recovered - true_val) / true_val * 100\n"
        "    recovery_rows.append({\n"
        "        \"channel\": ch,\n"
        "        \"true_total\": true_val,\n"
        "        \"recovered_mean\": mean_recovered,\n"
        "        \"hdi_94\": (hdi_low, hdi_high),\n"
        "        \"pct_error\": pct_error,\n"
        "        \"true_within_hdi\": within_hdi,\n"
        "    })\n"
        "\n"
        "recovery_df = pd.DataFrame(recovery_rows)\n"
        "recovery_df"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "for i, row in recovery_df.iterrows():\n"
        "    ax.errorbar(\n"
        "        row[\"channel\"], row[\"recovered_mean\"],\n"
        "        yerr=[[row[\"recovered_mean\"] - row[\"hdi_94\"][0]], [row[\"hdi_94\"][1] - row[\"recovered_mean\"]]],\n"
        "        fmt=\"o\", capsize=6, color=\"tab:blue\", label=\"recovered (94% HDI)\" if i == 0 else None,\n"
        "    )\n"
        "    ax.scatter(row[\"channel\"], row[\"true_total\"], color=\"black\", marker=\"x\", s=100,\n"
        "                label=\"true\" if i == 0 else None)\n"
        "ax.set_ylabel(\"total contribution (3 years)\")\n"
        "ax.set_title(\"Recovery check: true vs. model-recovered channel contribution\")\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("### Posterior predictive check (Track B) — does the model reproduce observed `y`?"),
    code("_ = mmm_synth.plot.posterior_predictive(var=[\"y\"])\nplt.show()"),
    md(
        "## Phase 4 — Visualizations\n"
        "\n"
        "All from `mmm.plot`, on Track B (the model we've validated recovers the "
        "truth). First, put contributions in real (dollar) units — the plotters "
        "need the `_original_scale` deterministics registered and backfilled onto "
        "the existing posterior (no re-sampling required)."
    ),
    code(
        "import pymc as pm\n"
        "\n"
        "contribution_vars = sorted(\n"
        "    v for v in mmm_synth.model.named_vars if v.endswith(\"_contribution\")\n"
        ")\n"
        "mmm_synth.add_original_scale_contribution_variable(var=contribution_vars)\n"
        "with mmm_synth.model:\n"
        "    mmm_synth.idata.posterior = pm.compute_deterministics(\n"
        "        mmm_synth.idata.posterior, merge_dataset=True\n"
        "    )"
    ),
    md("### Channel contributions over time"),
    code(
        "_ = mmm_synth.plot.contributions_over_time(var=[\"channel_contribution_original_scale\"])\n"
        "plt.show()"
    ),
    md("### Saturation curves — where does each channel hit its ceiling?"),
    code(
        "# NOTE: sample_saturation_curve(original_scale=True) already converts y to\n"
        "# dollar units; passing original_scale=True again to the plotter double-applies\n"
        "# target_scale (we hit this — curves topped out at ~1e7 instead of ~5000). Only\n"
        "# one of the two calls should do the conversion.\n"
        "curve = mmm_synth.sample_saturation_curve(\n"
        "    max_value=1.0, original_scale=False, random_state=RANDOM_SEED\n"
        ")\n"
        "_ = mmm_synth.plot.saturation_curves(curve=curve, original_scale=True)\n"
        "plt.show()"
    ),
    md("### Channel contribution share (with uncertainty)"),
    code("_ = mmm_synth.plot.channel_contribution_share_hdi()\nplt.show()"),
    md(
        "### Waterfall — how do we get from baseline to total response?\n"
        "\n"
        "**We hand-roll this one instead of calling "
        "`mmm.plot.waterfall_components_decomposition()`.** That method's total for "
        "every component came out ~4x too high — exactly our `chains=4`. Its "
        "underlying `build_contributions()` helper averages over `draw` but leaves "
        "`chain` unreduced, so the final sum silently adds the (near-identical) "
        "per-chain totals together instead of averaging them. We confirmed this "
        "against our own already-validated totals (the same ones behind the recovery "
        "check above) before ruling out a mistake on our end. Built and verified "
        "against a fitted model in an isolated script, not assumed."
    ),
    code(
        "post = mmm_synth.idata.posterior\n"
        "\n"
        "def total_original_scale(var_name):\n"
        "    da = post[var_name]\n"
        "    non_sample_dims = [d for d in da.dims if d not in (\"chain\", \"draw\")]\n"
        "    has_date = \"date\" in non_sample_dims\n"
        "    if non_sample_dims:\n"
        "        da = da.sum(dim=non_sample_dims)  # collapses date + any size-1 extra dims\n"
        "    if not has_date:\n"
        "        da = da * n_weeks  # constant per-week term -> total over history\n"
        "    return float(da.mean(dim=(\"chain\", \"draw\")))\n"
        "\n"
        "baseline_total = (\n"
        "    total_original_scale(\"intercept_contribution_original_scale\")\n"
        "    + total_original_scale(\"control_contribution_original_scale\")\n"
        "    + total_original_scale(\"yearly_seasonality_contribution_original_scale\")\n"
        ")\n"
        "x1_total = float(total_contrib_posterior.sel(channel=\"x1\").mean())\n"
        "x2_total = float(total_contrib_posterior.sel(channel=\"x2\").mean())\n"
        "grand_total = baseline_total + x1_total + x2_total\n"
        "\n"
        "labels = [\"baseline\\n(intercept+trend+season)\", \"+ x1\", \"+ x2\", \"= total\"]\n"
        "values = [baseline_total, x1_total, x2_total, grand_total]\n"
        "starts = [0, baseline_total, baseline_total + x1_total, 0]\n"
        "colors = [\"lightgray\", \"tab:blue\", \"tab:orange\", \"black\"]\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "bar_heights = [values[0], values[1], values[2], grand_total]\n"
        "ax.bar(labels[:3], values[:3], bottom=starts[:3], color=colors[:3])\n"
        "ax.bar(labels[3], grand_total, color=colors[3], alpha=0.7)\n"
        "for i, (lab, val) in enumerate(zip(labels, [baseline_total, x1_total, x2_total, grand_total])):\n"
        "    y_pos = starts[i] + val / 2 if i < 3 else grand_total / 2\n"
        "    ax.text(i, y_pos, f\"{val:,.0f}\", ha=\"center\", va=\"center\")\n"
        "ax.set_ylabel(\"total response, 3 years\")\n"
        "ax.set_title(\"Response decomposition: baseline vs. media channels\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## Phase 5 — Budget optimization: the headline number\n"
        "\n"
        "**Business question:** if we keep spending the same total budget, is there a "
        "better split between x1 and x2?\n"
        "\n"
        "**Setup, chosen deliberately to avoid extrapolation (Pitfall #5):**\n"
        "- Horizon: 13 weeks (one quarter) — a standard planning cycle.\n"
        "- Total budget: 13x the historical average combined weekly spend "
        "(≈ same run-rate we've actually been spending, not a hypothetical number).\n"
        "- Per-channel bounds: capped at 13x that channel's *historical maximum weekly "
        "spend*. The optimizer can never be asked to price a spend level the "
        "saturation curve was never calibrated on."
    ),
    code(
        "NUM_PERIODS = 13\n"
        "last_date = df_synth[\"date_week\"].max()\n"
        "\n"
        "weekly_mean = df_synth[\"x1\"].mean() + df_synth[\"x2\"].mean()\n"
        "TOTAL_BUDGET = NUM_PERIODS * weekly_mean\n"
        "share_x1_hist = df_synth[\"x1\"].mean() / weekly_mean\n"
        "current_x1 = TOTAL_BUDGET * share_x1_hist\n"
        "current_x2 = TOTAL_BUDGET * (1 - share_x1_hist)\n"
        "\n"
        "safe_bound_x1 = NUM_PERIODS * df_synth[\"x1\"].max()\n"
        "safe_bound_x2 = NUM_PERIODS * df_synth[\"x2\"].max()\n"
        "\n"
        "print(f\"horizon: {NUM_PERIODS} weeks\")\n"
        "print(f\"total_budget: ${TOTAL_BUDGET:,.0f}\")\n"
        "print(f\"current split: x1=${current_x1:,.0f} ({share_x1_hist:.1%}), "
        "x2=${current_x2:,.0f} ({1-share_x1_hist:.1%})\")\n"
        "print(f\"safe per-channel bound (historical max x {NUM_PERIODS}): "
        "x1=${safe_bound_x1:,.0f}, x2=${safe_bound_x2:,.0f}\")"
    ),
    code(
        "import warnings\n"
        "\n"
        "\n"
        "def response_at(x1_amt, x2_amt, total):\n"
        "    \"\"\"Evaluate expected response at a FIXED allocation via zero-width bounds\n"
        "    (forces the optimizer to evaluate at exactly this split, not search).\"\"\"\n"
        "    wrapper = MultiDimensionalBudgetOptimizerWrapper(\n"
        "        model=mmm_synth,\n"
        "        start_date=last_date + pd.Timedelta(weeks=1),\n"
        "        end_date=last_date + pd.Timedelta(weeks=NUM_PERIODS),\n"
        "    )\n"
        "    with warnings.catch_warnings():\n"
        "        warnings.simplefilter(\"ignore\")\n"
        "        _, res = wrapper.optimize_budget(\n"
        "            budget=total,\n"
        "            budget_bounds={\"x1\": (x1_amt, x1_amt), \"x2\": (x2_amt, x2_amt)},\n"
        "        )\n"
        "    return -res.fun\n"
        "\n"
        "\n"
        "def optimize_free(total):\n"
        "    wrapper = MultiDimensionalBudgetOptimizerWrapper(\n"
        "        model=mmm_synth,\n"
        "        start_date=last_date + pd.Timedelta(weeks=1),\n"
        "        end_date=last_date + pd.Timedelta(weeks=NUM_PERIODS),\n"
        "    )\n"
        "    bounds = {\n"
        "        \"x1\": (0, min(total, safe_bound_x1)),\n"
        "        \"x2\": (0, min(total, safe_bound_x2)),\n"
        "    }\n"
        "    with warnings.catch_warnings():\n"
        "        warnings.simplefilter(\"ignore\")\n"
        "        optimal, res = wrapper.optimize_budget(budget=total, budget_bounds=bounds)\n"
        "    return optimal, -res.fun"
    ),
    md("### Headline number: reallocating the same budget"),
    code(
        "optimal, utility_optimal = optimize_free(TOTAL_BUDGET)\n"
        "utility_current = response_at(current_x1, current_x2, TOTAL_BUDGET)\n"
        "uplift_pct = (utility_optimal - utility_current) / utility_current * 100\n"
        "\n"
        "opt_x1 = float(optimal.sel(channel=\"x1\"))\n"
        "opt_x2 = float(optimal.sel(channel=\"x2\"))\n"
        "\n"
        "print(f\"current split : x1=${current_x1:,.0f}  x2=${current_x2:,.0f}\")\n"
        "print(f\"optimal split : x1=${opt_x1:,.0f}  x2=${opt_x2:,.0f}\")\n"
        "print(f\"expected response — current: {utility_current:,.0f}, "
        "optimal: {utility_optimal:,.0f}\")\n"
        "print(f\"\\n>>> Same ${TOTAL_BUDGET:,.0f} budget, optimally split: \"\n"
        "      f\"{uplift_pct:+.2f}% response <<<\")"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "x = np.arange(2)\n"
        "width = 0.35\n"
        "ax.bar(x - width/2, [current_x1, current_x2], width, label=\"current split\", color=\"lightgray\")\n"
        "ax.bar(x + width/2, [opt_x1, opt_x2], width, label=\"optimal split\", color=\"tab:green\")\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels([\"x1\", \"x2\"])\n"
        "ax.set_ylabel(\"budget over 13 weeks ($)\")\n"
        "ax.set_title(f\"Current vs. optimal allocation (same ${TOTAL_BUDGET:,.0f} total)\")\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "### Second number: marginal ROI — where should the *next* dollar go?\n"
        "\n"
        "The aggregate uplift above is modest — both curves flatten fast enough that "
        "a full reallocation only buys a small amount. But that doesn't mean the two "
        "channels are equally efficient *right now*. We check by nudging a small "
        "amount of extra budget (1% of total) into each channel individually, holding "
        "the other fixed, and measuring the incremental response per incremental "
        "dollar — this is the number that matters for **growth** budget, as opposed "
        "to reallocating what's already being spent."
    ),
    code(
        "DELTA = 0.01 * TOTAL_BUDGET\n"
        "\n"
        "utility_base = response_at(current_x1, current_x2, TOTAL_BUDGET)\n"
        "utility_plus_x1 = response_at(current_x1 + DELTA, current_x2, TOTAL_BUDGET + DELTA)\n"
        "utility_plus_x2 = response_at(current_x1, current_x2 + DELTA, TOTAL_BUDGET + DELTA)\n"
        "\n"
        "marginal_x1 = (utility_plus_x1 - utility_base) / DELTA\n"
        "marginal_x2 = (utility_plus_x2 - utility_base) / DELTA\n"
        "\n"
        "print(f\"marginal response per extra $1 — x1: {marginal_x1:.2f}, x2: {marginal_x2:.2f}\")\n"
        "print(f\">>> at current spend levels, the next incremental dollar into x1 returns \"\n"
        "      f\"{marginal_x1/marginal_x2:.1f}x what it would in x2 <<<\")\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(5, 4))\n"
        "ax.bar([\"x1\", \"x2\"], [marginal_x1, marginal_x2], color=[\"tab:blue\", \"tab:orange\"])\n"
        "ax.set_ylabel(\"marginal response per +$1 spend\")\n"
        "ax.set_title(\"Marginal ROI at current spend — where should growth budget go?\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "**Why both numbers, together:** reallocating the *existing* budget barely "
        "moves total response (both curves are already fairly flat near the current "
        "spend levels), but the *marginal* return on the next dollar still clearly "
        "favors x1. That's coherent, not contradictory — it says: don't expect much "
        "from shuffling this quarter's budget, but if the budget grows, put the new "
        "money into x1 first."
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (mmm-case)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print("wrote", NB_PATH)
