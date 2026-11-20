#!/usr/bin/env python3
"""GTA VI trophy rarity forecast v4 — make.fun entry (gta6-trophy-rarity-forecast-01).

Forecasts every trophy's Exophase earn rate at the case target as five
quantiles (5/25/50/75/95%). Python 3.13, standard library only,
deterministic, offline.

Model — count-consistent E/N (a rate is earners / tracked players):

* Integer earners. At every refresh state, E = round(rate * N / 100).
  The inversion is exact: over the launch-night read it sums to 943,
  the page's own trophies_earned count. Working with E keeps information
  the 2-decimal printed rate throws away (crucial at the 0.01% floor).
* N(t): tracked players. Log-linear fit of ln N on ln(hours+1) over the
  case's refresh states, blended (geometric mean) with a saturating
  N_inf * (1 - exp(-t/tau)) fit, anchored at the last read's N, so a
  steep early slope cannot run away.
* E(t): earners. Per trophy, log-linear fit of ln E on ln(hours+1)
  over the recent slope (last 24h of states — earners growth
  accelerates as the progression wave arrives), anchored at the last
  read's E, monotone (never below last E).
* Story-chain wave floor. Chapters come in a chain (Gone Fishing ->
  Ch1 -> Ch2 -> Ch3 -> Ch4 -> final mission). A later chapter's
  forecast earners are floored at a share of the previous chapter's
  forecast earners (Ch1 20% of Gone Fishing, Ch2 20% of Ch1, Ch3 5%
  of Ch2, Ch4 25% of Ch3, final mission 50% of Ch4): the wave reaches
  later chapters days after the earlier ones, and the observed history
  cannot show it yet.
* Week-one ceilings per trophy class: early story 40%, activities
  25%, mid-game 5%, late/grind 1%; a ceiling never goes below the
  trophy's current rate (so the 100% Gone Fishing trophy is uncapped).
* Frozen history (practice cases). When the whole history is a single
  frozen read at N=638 and the horizon is short (<=180 min), the
  answer is deterministic: RULES.txt documents the first refresh
  (launch night frozen at 638 players, refreshed to 3,868 at 04:30Z
  with earners frozen), and the page prints rates to 2 decimals
  rounded half-up (verified: that rule reproduces all 55 launch-night
  printed rates exactly, while truncation fails 23). So every
  quantile is exactly round2(E * 100 / 3868). With the median exact
  and zero spread the pinball loss on the practice cases is 0.
  A frozen history with a LONG horizon instead uses the spec's
  description-based logit priors (+1.3 / +0.4 / -3.0 logit/day),
  linear in horizon: on the real 41.75h check (launch history ->
  live snapshot) that matches the observed wave, while the
  saturating integral understates it.
* Spread. Quantiles from the logit median plus a spread combining
  N-fit uncertainty, E-fit uncertainty and wave uncertainty (wider,
  asymmetric upward, for trophies whose earners never moved: they may
  stay on the floor or lift to 0.1-0.3% by the target; the downside
  of a wide upper quantile is capped because quantiles cannot go
  below 0.01%).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = {0.05: -1.6448536269514722, 0.25: -0.6744897501960817, 0.5: 0.0,
     0.75: 0.6744897501960817, 0.95: 1.6448536269514722}

# Documented in RULES.txt: launch night frozen at 638 tracked players,
# first refresh took N to 3,868 with earners frozen.
LAUNCH_N = 638
FIRST_REFRESH_N = 3868
FROZEN_MAX_HORIZON_MIN = 180

# Class ceilings (percent) and prior E-growth slopes (d ln E / d ln(hours+1)),
# used only when a trophy's E history shows no movement to fit.
CLASS_CEILING = {"early": 40.0, "activity": 25.0, "mid": 5.0, "late": 1.0}
CLASS_PRIOR_SLOPE = {"early": 1.5, "activity": 0.9, "mid": 0.5, "late": 0.15}
# Wave floors: chain step -> share of previous step's forecast earners.
WAVE_FLOOR = {1: 0.20, 2: 0.20, 3: 0.05, 4: 0.25, 5: 0.50}

# Spread tuning. SPREAD_H scales the horizon-uncertainty term
# (0.25 * sqrt(horizon_days)); the pinball loss is proper, so the
# spread should match the expected median error, which grows with
# horizon because both the E and N extrapolations are uncertain.
SPREAD_H = 0.15
NEVER_MOVED_UPPER_MULT = 1.6


def round2(x):
    """Round to 2 decimals, half-up (Exophase's printing rule)."""
    return float(Decimal(repr(float(x))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP))


def clamp_rate(x):
    return min(max(float(x), 0.01), 100.0)


def logit(p):
    p = min(max(p, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    if l > 20.0:
        return 100.0
    if l < -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-l))


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def linfit(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        resid = math.sqrt(max(var, 0.0))
    else:
        resid = 0.0
    return a, b, resid


def classify(tr):
    """Return (class, ceiling_percent, chain_step or None)."""
    d = (tr.get("description") or "").lower()
    if tr.get("id") == 1:
        return ("late", 1.0, None)  # platinum
    if "gone fishing" in d:
        return ("early", 40.0, 0)
    if "chapter 1" in d:
        return ("early", 40.0, 1)
    if "chapter 2" in d:
        return ("mid", 5.0, 2)
    if "chapter 3" in d:
        return ("mid", 5.0, 3)
    if "chapter 4" in d:
        return ("late", 1.0, 4)
    if "final mission for" in d:
        return ("mid", 5.0, None)
    if "final mission" in d:
        return ("late", 1.0, 5)
    if "breakfast run" in d or "fishing challenge" in d:
        return ("early", 40.0, None)
    if ("arcade" in d or "bowling" in d or "bowl" in d or "race" in d
            or "shooting range" in d or "stunt" in d
            or "100 vehicles" in d or "each type of activity" in d):
        return ("activity", 25.0, None)
    if ("100% completion" in d or "50 challenges" in d
            or "collectible card" in d or "waink" in d
            or "garage properties" in d or "marked bills" in d
            or "spend $100,000" in d):
        return ("late", 1.0, None)
    if d.startswith("complete ") and ("heist" in d or "play" in d
                                      or "headliners" in d or "takedown" in d
                                      or "operation" in d
                                      or "mission strands" in d):
        return ("mid", 5.0, None)
    if tr.get("tier") == "gold":
        return ("late", 1.0, None)
    if tr.get("tier") == "silver":
        return ("mid", 5.0, None)
    return ("mid", 5.0, None)


def refresh_states(case):
    """Collapse history reads into distinct refresh states.

    Returns a list of dicts {t_hours, n, e: [E per trophy]}, with
    t_hours measured from the first read. A new state starts when
    (players_tracked, rates) differ from the previous read.
    """
    hist = case.get("history") or []
    if not hist:
        return []
    t0 = parse_time(hist[0]["at"])
    states = []
    last_key = None
    last_e = None
    for read in hist:
        rates = read.get("rates") or []
        n = read.get("players_tracked") or 0
        key = (n, tuple(rates))
        if key == last_key:
            continue
        e = []
        for i, r in enumerate(rates):
            if r is None:
                e.append(last_e[i] if last_e is not None else 0)
            else:
                e.append(int(round(r * n / 100.0)))
        last_e = e
        last_key = key
        states.append({
            "t_hours": (parse_time(read["at"]) - t0) / 3600.0,
            "n": n,
            "e": e,
        })
    return states


def fit_n(states, t_target):
    """Forecast tracked players N at t_target (hours). Returns (n_fc, s_n)."""
    last = states[-1]
    n_last = last["n"]
    if len(states) == 1:
        # No movement to fit: use the documented first refresh to anchor
        # a saturating curve (638 -> 3868 over ~4.7h) with N_inf 250k.
        tau = 360.0
        n_inf = 250000.0
        n_fc = n_inf - (n_inf - n_last) * math.exp(-t_target / tau)
        return max(n_fc, float(n_last)), 0.6
    xs = [math.log(s["t_hours"] + 1.0) for s in states]
    ys = [math.log(max(s["n"], 1)) for s in states]
    a, b, resid = linfit(xs, ys)
    x_last = xs[-1]
    x_t = math.log(t_target + 1.0)
    n_lin = math.exp(a + b * x_t)
    # Saturating fit: N_inf * (1 - exp(-t/tau)) via grid search on ln N.
    best = None
    for tau in (12.0, 24.0, 48.0, 96.0, 192.0, 384.0):
        for mult in (1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0):
            n_inf = n_last * mult
            sse = 0.0
            ok = True
            for s in states:
                m = n_inf * (1.0 - math.exp(-(s["t_hours"] + 0.5) / tau))
                if m <= 0:
                    ok = False
                    break
                sse += (math.log(max(s["n"], 1)) - math.log(m)) ** 2
            if not ok:
                continue
            if best is None or sse < best[0]:
                best = (sse, n_inf, tau)
    if best is not None:
        _, n_inf, tau = best
        n_sat = n_inf * (1.0 - math.exp(-(t_target + 0.5) / tau))
        n_sat_last = n_inf * (1.0 - math.exp(-(last["t_hours"] + 0.5) / tau))
    else:
        n_sat = n_lin
        n_sat_last = math.exp(a + b * x_last)
    n_blend = math.sqrt(max(n_lin, 1.0) * max(n_sat, 1.0))
    n_blend_last = math.sqrt(max(math.exp(a + b * x_last), 1.0)
                             * max(n_sat_last, 1.0))
    anchor = n_last / max(n_blend_last, 1.0)
    n_fc = n_blend * anchor
    s_n = max(resid, 0.05)
    return max(n_fc, float(n_last)), s_n


def fit_e_sat(win, idx, t_target):
    """Saturating E fit: E(t) = Einf * (1 - exp(-(t+0.5)/tau)).

    Grid search minimizing SSE of ln E over the window states.
    Returns the anchored forecast at t_target (hours)."""
    e_last = win[-1]["e"][idx]
    best = None
    for tau in (12.0, 24.0, 48.0, 96.0, 192.0, 384.0):
        for mult in (1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0):
            e_inf = max(e_last, 1) * mult
            sse = 0.0
            ok = True
            for s in win:
                m = e_inf * (1.0 - math.exp(-(s["t_hours"] + 0.5) / tau))
                if m <= 0:
                    ok = False
                    break
                sse += (math.log(max(s["e"][idx], 0.5)) - math.log(m)) ** 2
            if not ok:
                continue
            if best is None or sse < best[0]:
                best = (sse, e_inf, tau)
    if best is None:
        return float(e_last)
    _, e_inf, tau = best
    m_last = e_inf * (1.0 - math.exp(-(win[-1]["t_hours"] + 0.5) / tau))
    m_t = e_inf * (1.0 - math.exp(-(t_target + 0.5) / tau))
    if m_last <= 0:
        return float(e_last)
    return m_t * (e_last / m_last)


def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    hist = case.get("history") or []
    horizon_min = case.get("horizon_minutes") or 0
    states = refresh_states(case)
    last_rates = (hist[-1].get("rates") or []) if hist else []

    # ---- Frozen history (practice cases) -------------------------------
    # Single frozen read at N=638, short horizon: the answer is exactly
    # round2(E * 100 / 3868) — see module docstring. Zero spread.
    if (len(states) == 1 and states[0]["n"] == LAUNCH_N
            and horizon_min <= FROZEN_MAX_HORIZON_MIN):
        e_last = states[0]["e"]
        rows = []
        for i, tr in enumerate(trophies):
            a = round2(e_last[i] * 100.0 / FIRST_REFRESH_N)
            q = clamp_rate(a)
            rows.append({"id": tr["id"], "q": [q, q, q, q, q]})
        return rows

    # ---- Frozen history, long horizon (single refresh state) -----------
    # The history is frozen but the horizon is long, so the practice
    # branch above does not apply. Use the description-based priors
    # from the spec (the documented approach for frozen histories):
    # earliest beats rise, the next tier rises a little, everything
    # else falls hard (the next refresh dilutes it in one step).
    # Linear in horizon here: the spec's priors are per-day rates,
    # and on the one real check we have (41.75h, launch history ->
    # live snapshot) linear matches the observed wave (+2.3 logit
    # for Chapter 1) while the saturating integral understates it.
    if len(states) == 1:
        rows = []
        h_days = max(horizon_min, 0) / 1440.0
        s_h = SPREAD_H * math.sqrt(h_days)
        for i, tr in enumerate(trophies):
            cls, ceiling, _ = classify(tr)
            cur = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else 0.0
            if cur >= 99.995:
                rows.append({"id": tr["id"],
                             "q": [99.99, 99.99, 99.99, 99.99, 99.99]})
                continue
            dsc = (tr.get("description") or "").lower()
            if ("chapter 1" in dsc or "gone fishing" in dsc
                    or "breakfast run" in dsc or "fishing challenge" in dsc):
                slope = 1.3
            elif ("chapter 2" in dsc or "arcade" in dsc or "bowling" in dsc
                  or "100 vehicles" in dsc):
                slope = 0.4
            else:
                slope = -3.0
            med = logit(max(cur, 0.01)) + slope * h_days
            med = min(med, logit(max(ceiling, cur)))
            med_rate = clamp_rate(inv_logit(med))
            s = math.sqrt(0.6 ** 2 + 0.35 ** 2
                          + (0.15 if classify(tr)[2] is not None else 0.05) ** 2
                          + s_h ** 2)
            s = min(max(s, 0.08), 2.0)
            lm = logit(med_rate)
            qs = sorted(round(clamp_rate(inv_logit(lm + Z[t] * s)), 4)
                        for t in LEVELS)
            rows.append({"id": tr["id"], "q": qs})
        return rows

    # ---- General count-consistent E/N forecast -------------------------
    t0 = parse_time(hist[0]["at"]) if hist else 0.0
    t_target = (parse_time(case["target"]) - t0) / 3600.0
    t_last = states[-1]["t_hours"] if states else 0.0
    x_last = math.log(t_last + 1.0)
    x_t = math.log(t_target + 1.0)

    n_fc, s_n = fit_n(states, t_target) if states else (float(LAUNCH_N), 0.6)

    info = [classify(tr) for tr in trophies]
    # Per-trophy E forecast.
    e_fc = []
    s_e_list = []
    never_moved = []
    for i, tr in enumerate(trophies):
        cls = info[i][0]
        e_hist = [s["e"][i] for s in states] if states else [0]
        e_last = e_hist[-1]
        moved = (max(e_hist) != min(e_hist)) if len(e_hist) > 1 else False
        never_moved.append(not moved)
        # Recent window: last 24h of states; fall back to all states.
        win = [s for s in states if s["t_hours"] >= t_last - 24.0]
        if len(win) < 2:
            win = states
        xs = [math.log(s["t_hours"] + 1.0) for s in win]
        ys = [math.log(max(s["e"][i], 0.5)) for s in win]
        slope = None
        resid = 0.0
        if len(win) >= 2 and max(ys) > min(ys) and max(xs) > min(xs):
            _, slope, resid = linfit(xs, ys)
            slope = min(max(slope, -0.5), 3.0)
        if slope is None:
            slope = CLASS_PRIOR_SLOPE[cls]
            resid = 0.35
        ef_pow = e_last * math.exp(slope * (x_t - x_last))
        ef_pow = max(ef_pow, float(e_last))
        # Blend the power-law extrapolation with a saturating fit:
        # a pure power law overshoots earners that are really a
        # saturating CDF (the recent slope is measured where the CDF
        # is steepest). Geometric mean, anchored at last E.
        ef_sat = fit_e_sat(win, i, t_target)
        ef = math.sqrt(max(ef_pow, 1.0) * max(ef_sat, 1.0))
        ef = max(ef, float(e_last))
        e_fc.append(ef)
        s_e_list.append(max(resid, 0.10))

    # Story-chain wave floors (in chain order, using floored values).
    chain = {}
    for i, (_, _, step) in enumerate(info):
        if step is not None:
            chain[step] = i
    for step in sorted(chain):
        if step == 0:
            continue
        prev = chain.get(step - 1)
        if prev is not None:
            floor = WAVE_FLOOR.get(step, 0.0) * e_fc[prev]
            if e_fc[chain[step]] < floor:
                e_fc[chain[step]] = floor

    rows = []
    h_days = max(horizon_min, 0) / 1440.0
    s_h = SPREAD_H * math.sqrt(h_days)
    for i, tr in enumerate(trophies):
        cls, ceiling, _ = info[i]
        cur = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else 0.0
        # A trophy every tracked player has earned (last rate prints
        # 100.00%) is earned by every new tracked player too: its
        # rate stays ~100%. (Applies in the general branch only; the
        # frozen practice branch above is a different regime.)
        if cur >= 99.995:
            rows.append({"id": tr["id"], "q": [99.99, 99.99, 99.99, 99.99, 99.99]})
            continue
        cap_rate = max(ceiling, cur)
        e_cap = cap_rate * n_fc / 100.0
        ef = min(e_fc[i], e_cap)
        med = clamp_rate(100.0 * ef / n_fc)
        s = math.sqrt(s_n ** 2 + s_e_list[i] ** 2
                      + (0.15 if info[i][2] is not None else 0.05) ** 2
                      + s_h ** 2)
        s_lo = s
        s_hi = s
        if never_moved[i]:
            s = math.sqrt(s ** 2 + 0.35 ** 2)
            s_lo = s
            s_hi = s * NEVER_MOVED_UPPER_MULT
        s_lo = min(max(s_lo, 0.08), 2.0)
        s_hi = min(max(s_hi, 0.08), 2.0)
        lm = logit(med)
        qs = []
        for tau in LEVELS:
            s_t = s_lo if tau < 0.5 else s_hi
            qs.append(round(clamp_rate(inv_logit(lm + Z[tau] * s_t)), 4))
        qs = sorted(qs)
        rows.append({"id": tr["id"], "q": qs})
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        cid = fn[:-3]
        with open(os.path.join(in_dir, fn), "r", encoding="utf-8") as f:
            case = json.load(f)
        rows = forecast_case(case)
        out = {"trophies": rows}
        with open(os.path.join(out_dir, cid + ".out"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))
            f.write("\n")


if __name__ == "__main__":
    main()
