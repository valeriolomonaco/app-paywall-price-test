"""Build the one-page website docs/index.html from the simulated data.

Run from the repository root, after data/simulate.py:

    python tools/build_site.py

Every number and chart on the page is computed here from data/users.csv and from
repeat runs of the simulation, so the page cannot drift from the notebook.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
import simulate  # noqa: E402

ALPHA, POWER, P_A, P_B = 0.05, 0.80, 0.050, 0.046

# ---- numbers -----------------------------------------------------------------------------

users = pd.read_csv(ROOT / "data" / "users.csv", parse_dates=["exposure_date"])
A, B = users[users.group == "A"], users[users.group == "B"]
N = len(A)
rev_a, rev_b = A.revenue_12m.to_numpy(), B.revenue_12m.to_numpy()


def users_per_group(p1, p2):
    z = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)
    return math.ceil(z * z * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2)


def boot(values, rng, repeats=5000, block=100):
    out = np.empty(repeats)
    for start in range(0, repeats, block):
        out[start:start + block] = values[rng.integers(0, len(values), size=(block, len(values)))].mean(axis=1)
    return out


rng = np.random.default_rng(1)
b_ra, b_rb = boot(rev_a, rng), boot(rev_b, rng)
b_ca, b_cb = boot(A.paid.to_numpy(float), rng), boot(B.paid.to_numpy(float), rng)
conv_a, conv_b = A.paid.mean(), B.paid.mean()
conv_lift, rev_lift = conv_b / conv_a - 1, rev_b.mean() / rev_a.mean() - 1
conv_ci = np.percentile(b_cb / b_ca - 1, [2.5, 97.5])
rev_ci = np.percentile(b_rb / b_ra - 1, [2.5, 97.5])
p_rev = ttest_ind(rev_b, rev_a, equal_var=False).pvalue
pooled = (A.paid.sum() + B.paid.sum()) / (2 * N)
p_conv = 2 * norm.sf(abs((conv_b - conv_a) / math.sqrt(pooled * (1 - pooled) * 2 / N)))
months_a, months_b = A.months_paid.sum() / A.paid.sum(), B.months_paid.sum() / B.paid.sum()
arppu_lift = (rev_b.sum() / B.paid.sum()) / (rev_a.sum() / A.paid.sum()) - 1
z_total = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)
var_sum = rev_a.var(ddof=1) + rev_b.var(ddof=1)
detectable = z_total * math.sqrt(var_sum / N) / rev_a.mean()
confirm_n = math.ceil(z_total ** 2 * var_sum / (rev_lift * rev_a.mean()) ** 2)

# Bayesian reading: weak priors, exact posteriors (same model as part 6 of the notebook)
MONTH_VALUES = np.arange(1, 13)


def summarise(g):
    return {"n": len(g), "payers": int(g.paid.sum()),
            "months": np.bincount(g.loc[g.paid == 1, "months_paid"], minlength=13)[1:], "price": g.price_eur.iloc[0]}


def posterior_revenue(c, rng, draws=200_000):
    conv = rng.beta(1 + c["payers"], 1 + c["n"] - c["payers"], size=draws)
    return conv * c["price"] * (rng.dirichlet(1 + c["months"], size=draws) @ MONTH_VALUES)


def classical_switch(c1, c2):
    def mean_var(c):
        mean = c["price"] * (c["months"] @ MONTH_VALUES) / c["n"]
        return mean, c["price"] ** 2 * (c["months"] @ MONTH_VALUES ** 2) / c["n"] - mean ** 2
    (m1, v1), (m2, v2) = mean_var(c1), mean_var(c2)
    return (m2 - m1) / math.sqrt(v1 / c1["n"] + v2 / c2["n"]) > norm.ppf(1 - ALPHA / 2)


def bayes_probability(c1, c2, rng, draws=4000):
    return (posterior_revenue(c2, rng, draws) > posterior_revenue(c1, rng, draws)).mean()


_rng = np.random.default_rng(3)
_pa, _pb = posterior_revenue(summarise(A), _rng), posterior_revenue(summarise(B), _rng)
p_bayes = (_pb > _pa).mean()
post_lift = 100 * (_pb / _pa - 1)
post_edges = np.arange(-10, 27, 1)
post_hist, _ = np.histogram(post_lift, bins=post_edges)
cred = np.percentile(post_lift, [2.5, 97.5])
loss_switch, loss_stay = 1e5 * np.maximum(_pa - _pb, 0).mean(), 1e5 * np.maximum(_pb - _pa, 0).mean()

# 300 repeats of the experiment
repeat_lift, world_1 = [], []
sim_rng = np.random.default_rng(5)
for seed in range(300):
    run = simulate.simulate(seed)
    a = run.loc[run.group == "A", "revenue_12m"].to_numpy()
    b = run.loc[run.group == "B", "revenue_12m"].to_numpy()
    repeat_lift.append(b.mean() / a.mean() - 1)
    c1, c2 = summarise(run[run.group == "A"]), summarise(run[run.group == "B"])
    world_1.append((classical_switch(c1, c2), bayes_probability(c1, c2, sim_rng)))
repeat_lift = np.array(repeat_lift)

# 2,000 tests in which both groups pay EUR 4.99 and behave alike: any switch is a false alarm
churn = simulate.MONTHLY_CHURN["A"]
month_shares = np.array([(1 - churn) ** (k - 1) * churn for k in range(1, 12)] + [(1 - churn) ** 11])
world_2 = []
for _ in range(2000):
    pair = []
    for _ in range(2):
        payers = sim_rng.binomial(N, simulate.PAID_RATE["A"])
        pair.append({"n": N, "payers": payers, "months": sim_rng.multinomial(payers, month_shares), "price": simulate.PRICE["A"]})
    world_2.append((classical_switch(*pair), bayes_probability(*pair, sim_rng)))
fa_classical = 100 * np.mean([w[0] for w in world_2])
fa_bayes95 = 100 * np.mean([w[1] >= 0.95 for w in world_2])
fa_bayes975 = 100 * np.mean([w[1] >= 0.975 for w in world_2])

# open datasets (part 5 of the notebook). If the files are absent, the last saved results are used.
OPEN, OPEN_CACHE = ROOT / "data" / "open", ROOT / "tools" / "open_data_results.json"
if (OPEN / "AB_Test_Results.csv").exists() and (OPEN / "marketing_AB.csv").exists():
    raw = pd.read_csv(OPEN / "AB_Test_Results.csv")
    per_user = raw.groupby("USER_ID")["VARIANT_NAME"].nunique()
    in_both = per_user[per_user > 1].index
    clean = raw[~raw.USER_ID.isin(in_both)].groupby(["USER_ID", "VARIANT_NAME"], as_index=False).REVENUE.sum()
    ctl = clean.loc[clean.VARIANT_NAME == "control", "REVENUE"].to_numpy()
    var = clean.loc[clean.VARIANT_NAME == "variant", "REVENUE"].to_numpy()

    def open_compare(a, b, seed):
        r = np.random.default_rng(seed)
        ba, bb = boot(a, r), boot(b, r)
        lo, hi = np.percentile(bb / ba - 1, [2.5, 97.5])
        return [100 * (b.mean() / a.mean() - 1), 100 * lo, 100 * hi]

    top = max(ctl.max(), var.max())
    mk = pd.read_csv(OPEN / "marketing_AB.csv")
    ad, psa = mk.loc[mk["test group"] == "ad", "converted"], mk.loc[mk["test group"] == "psa", "converted"]
    open_res = {"rows": int(len(raw)), "users": int(raw.USER_ID.nunique()), "in_both": int(len(in_both)),
                "payers": [int((ctl > 0).sum()), int((var > 0).sum())],
                "all": open_compare(ctl, var, 11), "no_top": open_compare(ctl[ctl < top], var[var < top], 12),
                "mk_users": int(len(mk)), "mk_small_share": float(100 * len(psa) / len(mk)),
                "mk_conv": [float(100 * psa.mean()), float(100 * ad.mean())]}
    OPEN_CACHE.write_text(json.dumps(open_res, indent=1), encoding="utf-8")
else:
    open_res = json.loads(OPEN_CACHE.read_text(encoding="utf-8"))
edges = np.arange(4, 34, 2)
hist, _ = np.histogram(100 * repeat_lift, bins=edges)

# peeking: tests with no true difference
rng = np.random.default_rng(2)
SIMS, BLOCKS = 20_000, 20
block = N // BLOCKS
pa = rng.binomial(block, P_A, size=(SIMS, BLOCKS)).cumsum(axis=1)
pb = rng.binomial(block, P_A, size=(SIMS, BLOCKS)).cumsum(axis=1)
n_so_far = block * np.arange(1, BLOCKS + 1)
pooled_s = (pa + pb) / (2 * n_so_far)
z = (pb - pa) / n_so_far / np.sqrt(pooled_s * (1 - pooled_s) * 2 / n_so_far)
sig = np.abs(z) > norm.ppf(1 - ALPHA / 2)
peek = {k: 100 * sig[:, np.arange(1, k + 1) * BLOCKS // k - 1].any(axis=1).mean() for k in (1, 2, 5, 10, 20)}

# novelty: weekly conversion gap
users["day"] = (users.exposure_date - users.exposure_date.min()).dt.days
users["week"] = users.day // 7 + 1
weekly = []
for week, chunk in users.groupby("week"):
    a, b = chunk.loc[chunk.group == "A", "paid"], chunk.loc[chunk.group == "B", "paid"]
    ratio = b.mean() / a.mean()
    se = math.sqrt((1 - a.mean()) / a.sum() + (1 - b.mean()) / b.sum())
    weekly.append((week, 100 * (ratio - 1), 100 * (ratio * math.exp(-1.96 * se) - 1), 100 * (ratio * math.exp(1.96 * se) - 1)))
days = np.arange(simulate.TEST_DAYS)
built_in = 100 * ((simulate.PAID_RATE["B"] / simulate.PAID_RATE["A"]) * (
    1 + simulate.NOVELTY_BOOST * np.exp(-days / simulate.NOVELTY_DECAY_DAYS)) - 1).reshape(8, 7).mean(axis=1)
w1, rest = users[users.week == 1], users[users.week > 1]
gap_w1 = w1[w1.group == "B"].paid.mean() / w1[w1.group == "A"].paid.mean() - 1
gap_rest = rest[rest.group == "B"].paid.mean() / rest[rest.group == "A"].paid.mean() - 1

# ---- SVG helpers (720 x 400 canvas) --------------------------------------------------------

W, H, X0, X1, Y0, Y1 = 720, 480, 58, 704, 26, 418


def pct(v, digits=1):
    return f"{v:+.{digits}f}%".replace("-", "−")


def frame(body, label):
    return f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label}">{body}</svg>'


def y_axis(ymin, ymax, ticks, fmt=lambda v: f"{v:g}"):
    sy = lambda v: Y1 - (v - ymin) / (ymax - ymin) * (Y1 - Y0)  # noqa: E731
    out = "".join(
        f'<line class="grid" x1="{X0}" x2="{X1}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
        f'<text class="tick" x="{X0 - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{fmt(t)}</text>' for t in ticks)
    return sy, out


def bar_chart(cats, vals, ymin, ymax, ticks, highlight, labels, aria, hline=None, xlabel=None):
    sy, out = y_axis(ymin, ymax, ticks)
    slot = (X1 - X0) / len(cats)
    for i, (c, v) in enumerate(zip(cats, vals)):
        x, w = X0 + slot * i + slot * 0.09, slot * 0.82
        top, bottom = min(sy(v), sy(0)), max(sy(v), sy(0))
        fill = "var(--series-4)" if i == highlight else "var(--series-1)"
        grow = "grow" if v >= 0 else "grow down"
        out += f'<rect class="{grow}" style="--d:{0.15 + 0.12 * i:.2f}s;fill:{fill}" x="{x:.1f}" y="{top:.1f}" width="{w:.1f}" height="{bottom - top:.1f}"/>'
        ly = top - 8 if v >= 0 else bottom + 18
        strong = " strong" if i == highlight else ""
        out += f'<text class="label fade{strong}" style="--d:{0.55 + 0.12 * i:.2f}s" x="{x + w / 2:.1f}" y="{ly:.1f}" text-anchor="middle">{labels[i]}</text>'
        for j, line in enumerate(c.split("\n")):
            out += f'<text class="tick" x="{x + w / 2:.1f}" y="{Y1 + 20 + 15 * j:.1f}" text-anchor="middle">{line}</text>'
    out += f'<line x1="{X0}" x2="{X1}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" style="stroke:var(--navy)" stroke-width="1"/>'
    if hline is not None:
        out += f'<line x1="{X0}" x2="{X1}" y1="{sy(hline):.1f}" y2="{sy(hline):.1f}" style="stroke:var(--navy)" stroke-width="1" stroke-dasharray="6 5"/>'
    if xlabel:
        out += f'<text class="tick" x="{(X0 + X1) / 2}" y="{H - 6}" text-anchor="middle">{xlabel}</text>'
    return frame(out, aria)


def chart_sample():
    drops = np.arange(0.04, 0.2001, 0.005)
    need = np.array([users_per_group(P_A, P_A * (1 - d)) for d in drops]) / 1000
    sx = lambda v: X0 + (v - 4) / 16 * (X1 - X0)  # noqa: E731
    sy, out = y_axis(0, 200, [0, 50, 100, 150, 200])
    pts = " ".join(f"{sx(100 * d):.1f},{sy(n):.1f}" for d, n in zip(drops, need))
    out += "".join(f'<text class="tick" x="{sx(t):.1f}" y="{Y1 + 20}" text-anchor="middle">{t}%</text>' for t in range(4, 21, 2))
    out += f'<polyline class="draw" pathLength="1" points="{pts}" style="stroke:var(--series-1)" fill="none" stroke-width="4" stroke-linejoin="round"/>'
    n8 = users_per_group(P_A, P_B) / 1000
    out += f'<circle class="fade" style="--d:1.1s;fill:var(--series-4);stroke:var(--white)" stroke-width="3" cx="{sx(8):.1f}" cy="{sy(n8):.1f}" r="9"/>'
    out += f'<text class="label strong fade" style="--d:1.25s" x="{sx(8) + 18:.1f}" y="{sy(n8) - 26:.1f}">This test: 8% fall</text>'
    out += f'<text class="label fade" style="--d:1.25s" x="{sx(8) + 18:.1f}" y="{sy(n8) - 8:.1f}">{users_per_group(P_A, P_B):,} users per group</text>'
    out += f'<line class="axis" x1="{X0}" x2="{X1}" y1="{Y1}" y2="{Y1}"/>'
    out += f'<text class="tick" x="{(X0 + X1) / 2}" y="{H - 6}" text-anchor="middle">Smallest fall in conversion to detect</text>'
    out += f'<text class="tick" x="4" y="{Y0 - 10}">thousand</text>'
    return frame(out, "Users needed per group against the smallest fall in conversion to detect")


def chart_forest(rows=None, lo_x=-20, hi_x=20, step=5, xlabel="EUR 6.99 compared with EUR 4.99, with 95% interval",
                 aria="Conversion falls 8.1 percent with an interval below zero; revenue per user rises 7.0 percent with an interval that includes zero", digits=1):
    sx = lambda v: X0 + 150 + (v - lo_x) / (hi_x - lo_x) * (X1 - X0 - 150)  # noqa: E731
    out = ""
    for t in range(lo_x, hi_x + 1, step):
        out += f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{Y0}" y2="{Y1}"/><text class="tick" x="{sx(t):.1f}" y="{Y1 + 20}" text-anchor="middle">{t:+d}%</text>'.replace("+0%", "0%").replace("-", "−")
    if rows is None:
        rows = [("Conversion", "to paid", conv_lift, conv_ci, "var(--series-1)"), ("Revenue", "per user", rev_lift, rev_ci, "var(--series-4)")]
    for i, (l1, l2, v, ci, col) in enumerate(rows):
        y = Y0 + (Y1 - Y0) * (0.3 + 0.4 * i)
        lo, hi = sx(100 * ci[0]), sx(100 * ci[1])
        out += f'<text class="label strong" x="{X0}" y="{y - 2}">{l1}</text><text class="label" x="{X0}" y="{y + 16}">{l2}</text>'
        out += f'<rect class="growx" style="--d:{0.3 + 0.5 * i:.1f}s;fill:{col}" x="{lo:.1f}" y="{y - 3}" width="{hi - lo:.1f}" height="6"/>'
        out += f'<circle class="fade" style="--d:{0.9 + 0.5 * i:.1f}s;fill:{col};stroke:var(--white)" stroke-width="3" cx="{sx(100 * v):.1f}" cy="{y}" r="10"/>'
        out += f'<text class="label strong fade" style="--d:{0.9 + 0.5 * i:.1f}s;font-size:17px" x="{sx(100 * v):.1f}" y="{y - 22}" text-anchor="middle">{pct(100 * v, digits)}</text>'
    out += f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{Y0}" y2="{Y1}" style="stroke:var(--navy)" stroke-width="1.5" stroke-dasharray="6 5"/>'
    out += f'<text class="tick" x="{(X0 + 150 + X1) / 2}" y="{H - 6}" text-anchor="middle">{xlabel}</text>'
    return frame(out, aria)


def chart_posterior():
    sx = lambda v: X0 + (v + 10) / 37.5 * (X1 - X0)  # noqa: E731
    top = post_hist.max() * 1.18
    sy = lambda c: Y1 - c / top * (Y1 - Y0)  # noqa: E731
    out = ""
    for i, c in enumerate(post_hist):
        fill = "var(--series-1)" if post_edges[i] >= 0 else "var(--series-3)"
        x0, x1 = sx(post_edges[i]) + 0.5, sx(post_edges[i + 1]) - 0.5
        out += f'<rect class="grow" style="--d:{0.1 + 0.03 * i:.2f}s;fill:{fill}" x="{x0:.1f}" y="{sy(c):.1f}" width="{x1 - x0:.1f}" height="{Y1 - sy(c):.1f}"/>'
    out += "".join(f'<text class="tick" x="{sx(t):.1f}" y="{Y1 + 20}" text-anchor="middle">{t:+d}%</text>'.replace("+0%", "0%").replace("-", "−") for t in range(-10, 26, 5))
    out += f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{Y0}" y2="{Y1}" style="stroke:var(--navy)" stroke-width="1.5" stroke-dasharray="6 5"/>'
    out += f'<text class="label strong fade" style="--d:1.2s;fill:var(--blue)" x="{sx(13):.1f}" y="{Y0 + 60}">EUR 6.99 earns more</text>'
    out += f'<text class="label strong fade" style="--d:1.2s;fill:var(--blue);font-size:19px" x="{sx(13):.1f}" y="{Y0 + 84}">{100 * p_bayes:.1f}%</text>'
    out += f'<text class="label fade" style="--d:1.4s" x="{sx(-1):.1f}" y="{Y1 - 120}" text-anchor="end">EUR 4.99 earns more</text>'
    out += f'<text class="label fade" style="--d:1.4s;font-size:17px" x="{sx(-1):.1f}" y="{Y1 - 98}" text-anchor="end">{100 * (1 - p_bayes):.1f}%</text>'
    out += f'<line class="axis" x1="{X0}" x2="{X1}" y1="{Y1}" y2="{Y1}"/>'
    out += f'<text class="tick" x="{(X0 + X1) / 2}" y="{H - 6}" text-anchor="middle">Plausible revenue gain of EUR 6.99, per user</text>'
    return frame(out, "Posterior distribution of the revenue gain; about 97 percent of it lies above zero")


def chart_hist():
    sx = lambda v: X0 + (v - 4) / 28 * (X1 - X0)  # noqa: E731
    top = int(math.ceil(hist.max() / 20) * 20)
    sy, out = y_axis(0, top, list(range(0, top + 1, 20)))
    for i, c in enumerate(hist):
        x0, x1 = sx(edges[i]) + 1, sx(edges[i + 1]) - 1
        out += f'<rect class="grow" style="--d:{0.15 + 0.05 * i:.2f}s;fill:var(--series-1)" x="{x0:.1f}" y="{sy(c):.1f}" width="{x1 - x0:.1f}" height="{sy(0) - sy(c):.1f}"/>'
    out += "".join(f'<text class="tick" x="{sx(t):.1f}" y="{Y1 + 20}" text-anchor="middle">+{t}%</text>' for t in range(5, 31, 5))
    xm, xd = sx(100 * repeat_lift.mean()), sx(100 * rev_lift)
    out += f'<line class="fade" style="--d:1s;stroke:var(--navy)" x1="{xm:.1f}" x2="{xm:.1f}" y1="{Y0}" y2="{Y1}" stroke-width="1.5" stroke-dasharray="6 5"/>'
    out += f'<text class="label fade" style="--d:1s" x="{xm + 8:.1f}" y="{Y0 + 12}">Average of repeats {pct(100 * repeat_lift.mean())}</text>'
    out += f'<rect class="grow" style="--d:1.3s;fill:var(--series-4)" x="{xd - 3:.1f}" y="{Y0}" width="6" height="{Y1 - Y0}"/>'
    out += f'<text class="label strong fade" style="--d:1.6s;fill:var(--orange)" x="{xd + 10:.1f}" y="{Y0 + 44}">This dataset</text>'
    out += f'<text class="label strong fade" style="--d:1.6s;fill:var(--orange);font-size:17px" x="{xd + 10:.1f}" y="{Y0 + 64}">{pct(100 * rev_lift)}</text>'
    out += f'<line class="axis" x1="{X0}" x2="{X1}" y1="{Y1}" y2="{Y1}"/>'
    out += f'<text class="tick" x="{(X0 + X1) / 2}" y="{H - 6}" text-anchor="middle">Revenue gain of EUR 6.99 in each repeat</text>'
    out += f'<text class="tick" x="4" y="{Y0 - 10}">repeats</text>'
    return frame(out, "Histogram of the revenue gain in 300 repeat runs; this dataset lies below all of them")


def chart_novelty():
    sx = lambda w: X0 + (w - 0.5) / 8 * (X1 - X0)  # noqa: E731
    sy, out = y_axis(-35, 30, [-30, -20, -10, 0, 10, 20, 30], fmt=lambda v: f"{v:+d}%".replace("+0%", "0%").replace("-", "−"))
    for i, (w, g, lo, hi) in enumerate(weekly):
        d = 0.2 + 0.08 * i
        out += f'<rect class="growy" style="--d:{d:.2f}s;fill:var(--series-1)" x="{sx(w) - 1.5:.1f}" y="{sy(hi):.1f}" width="3" height="{sy(lo) - sy(hi):.1f}"/>'
        out += f'<circle class="fade" style="--d:{d + 0.2:.2f}s;fill:var(--series-1);stroke:var(--white)" stroke-width="2.5" cx="{sx(w):.1f}" cy="{sy(g):.1f}" r="8"/>'
        out += f'<text class="tick" x="{sx(w):.1f}" y="{Y1 + 20}" text-anchor="middle">{w}</text>'
    pts = " ".join(f"{sx(w + 1):.1f},{sy(v):.1f}" for w, v in enumerate(built_in))
    out += f'<polyline class="draw" style="--d:1s;stroke:var(--series-4)" pathLength="1" points="{pts}" fill="none" stroke-width="4" stroke-linejoin="round"/>'
    out += f'<line x1="{X0}" x2="{X1}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" style="stroke:var(--navy)" stroke-width="1" stroke-dasharray="6 5"/>'
    out += f'<rect x="{X1 - 250}" y="{Y0 + 2}" width="22" height="4" style="fill:var(--series-4)"/><text class="legend" x="{X1 - 222}" y="{Y0 + 9}">Built into the simulation</text>'
    out += f'<circle cx="{X1 - 239}" cy="{Y0 + 26}" r="6" style="fill:var(--series-1)"/><text class="legend" x="{X1 - 222}" y="{Y0 + 31}">This dataset, 95% interval</text>'
    out += f'<text class="tick" x="{(X0 + X1) / 2}" y="{H - 6}" text-anchor="middle">Week of the test</text>'
    return frame(out, "Weekly conversion gap between the groups, with the gap built into the simulation")


svg_sample, svg_forest, svg_hist, svg_novelty = chart_sample(), chart_forest(), chart_hist(), chart_novelty()
svg_posterior = chart_posterior()
svg_open = chart_forest(
    rows=[("All clean", "users", open_res["all"][0] / 100, [open_res["all"][1] / 100, open_res["all"][2] / 100], "var(--series-1)"),
          ("Without the", "largest payer", open_res["no_top"][0] / 100, [open_res["no_top"][1] / 100, open_res["no_top"][2] / 100], "var(--series-4)")],
    lo_x=-90, hi_x=40, step=30, xlabel="Revenue per user, variant compared with control, 95% interval",
    aria="Revenue per user in the open dataset: the interval is wide and includes zero, with or without the largest payer", digits=0)
svg_peek = bar_chart([f"{k} look" if k == 1 else f"{k} looks" for k in peek], list(peek.values()), 0, 30, [0, 10, 20, 30], 3,
                     [f"{v:.1f}%" for v in peek.values()], "False alarm rate by number of looks", hline=5,
                     xlabel="Looks at the result, stopping at the first significant one")
svg_metric = bar_chart(["Conversion\nto paid", "Revenue per\npaying user", "Revenue\nper user"],
                       [100 * conv_lift, 100 * arppu_lift, 100 * rev_lift], -12, 20, [-10, 0, 10, 20], 2,
                       [pct(100 * conv_lift), pct(100 * arppu_lift), pct(100 * rev_lift)],
                       "Change from EUR 4.99 to EUR 6.99 under three metrics")

# ---- page ----------------------------------------------------------------------------------

NUM = dict(
    n_total=f"{len(users):,}", n_group=f"{N:,}",
    conv_a=f"{100 * conv_a:.2f}", conv_b=f"{100 * conv_b:.2f}", conv_lift=pct(100 * conv_lift), conv_lift_raw=f"{100 * conv_lift:.1f}",
    rev_a=f"{rev_a.mean():.2f}", rev_b=f"{rev_b.mean():.2f}", rev_lift=pct(100 * rev_lift), rev_lift_raw=f"{100 * rev_lift:.1f}",
    rev_lo=pct(100 * rev_ci[0]), rev_hi=pct(100 * rev_ci[1]), p_rev=f"{p_rev:.3f}", p_conv=f"{p_conv:.3f}",
    months_a=f"{months_a:.1f}", months_b=f"{months_b:.1f}", months_lift=pct(100 * (months_b / months_a - 1)), months_lift_raw=f"{100 * (months_b / months_a - 1):.1f}",
    price_lift=pct(100 * (6.99 / 4.99 - 1)), price_lift_raw=f"{100 * (6.99 / 4.99 - 1):.1f}",
    arppu_lift=pct(100 * arppu_lift), rep_mean=pct(100 * repeat_lift.mean()), rep_mean_raw=f"{100 * repeat_lift.mean():.1f}",
    rep_min=pct(100 * repeat_lift.min()), rep_max=pct(100 * repeat_lift.max()),
    peek10=f"{peek[10]:.0f}", gap_w1=pct(100 * gap_w1), gap_rest=pct(100 * gap_rest),
    p_bayes=f"{100 * p_bayes:.0f}", p_bayes_raw=f"{100 * p_bayes:.1f}",
    cred_lo=pct(cred[0]).replace("−0.0%", "0.0%"), cred_hi=pct(cred[1]), loss_switch=f"{loss_switch:,.0f}", loss_stay=f"{loss_stay:,.0f}",
    loss_switch_raw=f"{loss_switch:.0f}", loss_stay_raw=f"{loss_stay:.0f}",
    one_sided=f"{100 * (1 - p_rev / 2):.1f}",
    fa_classical=f"{fa_classical:.1f}", fa_bayes95=f"{fa_bayes95:.1f}", fa_bayes975=f"{fa_bayes975:.1f}",
    open_rows=f"{open_res['rows']:,}", open_in_both=f"{open_res['in_both']:,}", open_in_both_raw=str(open_res["in_both"]),
    open_users=f"{open_res['users']:,}", open_payers_a=str(open_res["payers"][0]), open_payers_b=str(open_res["payers"][1]),
    open_all=pct(open_res["all"][0], 0), open_no_top=pct(open_res["no_top"][0], 0),
    mk_users=f"{open_res['mk_users']:,}", mk_small=f"{open_res['mk_small_share']:.0f}",
    mk_conv_a=f"{open_res['mk_conv'][0]:.2f}", mk_conv_b=f"{open_res['mk_conv'][1]:.2f}",
    svg_posterior=svg_posterior, svg_open=svg_open,
    detectable=pct(100 * detectable), confirm_n=f"{confirm_n:,}", share_b=f"{100 * (b_rb > b_ra).mean():.0f}",
    svg_sample=svg_sample, svg_forest=svg_forest, svg_hist=svg_hist, svg_peek=svg_peek, svg_novelty=svg_novelty, svg_metric=svg_metric,
)

template = (ROOT / "tools" / "site_template.html").read_text(encoding="utf-8")
for key, value in NUM.items():
    template = template.replace("{{" + key + "}}", value)
assert "{{" not in template, template[template.index("{{"):][:60]
(ROOT / "docs" / "index.html").write_text(template, encoding="utf-8", newline="\n")
print("wrote docs/index.html")
for k in ("conv_lift", "rev_lift", "rev_lo", "rev_hi", "p_rev", "arppu_lift", "rep_mean", "rep_min", "peek10", "gap_w1", "gap_rest", "detectable", "confirm_n", "share_b"):
    print(f"  {k}: {NUM[k]}".replace("−", "-"))
print("  histogram:", hist.tolist())
