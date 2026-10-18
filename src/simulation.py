"""
simulation.py
=============
A lost-sales, periodic-review single-echelon inventory simulator, vectorised
across series, plus the replenishment policies evaluated in the study.

Why lost sales: M5 is Walmart retail data, where unmet demand is abandoned
rather than backordered. Modelling it as backorders would systematically flatter
policies that under-stock.

Conventions
-----------
* Day index t is 0-based and refers to the global M5 calendar.
* A policy is given a target-level surface S of shape (n_days, n_series): the
  order-up-to level in force on day t. A periodic-review policy only reads S on
  its review days; a daily-review policy reads it every day.
* Lead time L is deterministic and identical for all series (sensitivities in
  the driver sweep L).
* Pipeline stock is tracked as a (n_series, L+1) arrival matrix, so on-hand and
  inventory position stay exact under lost sales.

Cost model (all quantities are per SKU-store-day unless noted)
-------------------------------------------------------------
  unit purchase cost  c_i = COST_RATIO * sell_price_i
  holding cost        h_i = HOLD_RATE * c_i / 365
  stockout penalty    p_i chosen so that the newsvendor critical ratio
                      p_i / (p_i + h_i) equals the target CR
  fixed order cost    K   per replenishment order placed for a series
"""
from __future__ import annotations

import numpy as np

COST_RATIO = 0.60      # purchase cost as a share of shelf price (40% gross margin)
HOLD_RATE = 0.25       # annual holding rate as a share of unit cost
BASE_CR = 0.90         # target critical ratio / cycle service level
BASE_K = 0.50         # fixed cost per SKU-store order line (USD)
BASE_LEAD = 2          # lead time (days)


def derive_costs(sell_price: np.ndarray, hold_rate: float = HOLD_RATE,
                 cost_ratio: float = COST_RATIO,
                 penalty_mode: str = "margin", penalty_mult: float = 1.0,
                 target_cr: float | None = None, review_days: int = 7):
    """
    Build the per-unit cost model.

        unit cost        c_i = cost_ratio * sell_price_i
        holding cost     h_i = hold_rate * c_i / 365      (per unit per DAY)
        stockout penalty p_i                              (per unit short, once)

    Two ways to pin down the penalty, both reported in the paper:

      penalty_mode="margin"  p_i = penalty_mult * (sell_price_i - c_i), i.e. the
                             forgone gross margin. Economically grounded and
                             needs no free parameter; the critical ratio is then
                             an OUTPUT of the cost model, not an input.
      penalty_mode="cr"      choose p_i so that the critical ratio equals
                             `target_cr` exactly. Lets us sweep service targets.

    Critical-ratio convention. Because the simulator charges holding per unit
    per day, the overage cost of raising the order-up-to level by one unit under
    a periodic-review base-stock policy is h_i * R, where R is the review
    interval in days (an unwanted unit is carried for about one cycle). The
    newsvendor optimality condition is therefore

        CR = p_i / (p_i + h_i * R)      [derivation in the paper's Appendix]

    and the matching quantile of protection-interval demand is the optimal
    order-up-to level. Using h_i instead of h_i * R -- an easy mistake --
    overstates the optimal service level substantially. Policies reviewed daily
    use R = 1 and therefore a different CR under the same economics, which is
    handled by `critical_ratio()`.
    """
    sp = np.maximum(np.nan_to_num(sell_price, nan=3.5), 0.01).astype(np.float64)
    c = cost_ratio * sp
    h = hold_rate * c / 365.0
    if penalty_mode == "margin":
        p = penalty_mult * (sp - c)
    elif penalty_mode == "cr":
        cr = float(np.clip(target_cr if target_cr is not None else BASE_CR,
                           1e-4, 1 - 1e-6))
        p = h * review_days * cr / (1.0 - cr)
    else:
        raise ValueError(penalty_mode)
    return (c.astype(np.float32), h.astype(np.float32), np.maximum(p, 1e-9).astype(np.float32))


def critical_ratio(h: np.ndarray, p: np.ndarray, review_days: int) -> np.ndarray:
    """CR = p / (p + h*R): the newsvendor quantile optimal for this policy's
    review frequency, given the shared economic cost model."""
    hR = np.asarray(h, dtype=np.float64) * float(review_days)
    return np.asarray(p, dtype=np.float64) / (np.asarray(p, dtype=np.float64) + hR)


def tau_index_for_cr(taus, cr) -> int:
    """Index of the closest available quantile level to a critical ratio."""
    return int(np.argmin(np.abs(np.asarray(taus, dtype=np.float64) - float(cr))))


# ==========================================================================
# simulator
# ==========================================================================
def simulate(demand: np.ndarray, S: np.ndarray, review_days: np.ndarray,
             lead: int = BASE_LEAD, h: np.ndarray | None = None,
             p: np.ndarray | None = None, K: float = BASE_K,
             init_on_hand: np.ndarray | None = None,
             min_order: float = 0.0) -> dict:
    """
    Run the inventory system.

    demand      (T, n)   realised daily unit demand inside the evaluation window
    S           (T, n)   order-up-to level in force each day
    review_days 1-D int  days (0-based within the window) on which orders may be placed
    lead        int      deterministic lead time in days
    h, p        (n,)     per-unit-per-day holding cost and per-unit stockout penalty
    K           float    fixed cost per order line placed
    init_on_hand (n,)    starting on-hand inventory; defaults to the day-0 target

    Returns a dict of daily traces and window-level cost / service statistics.
    """
    T, n = demand.shape
    demand = demand.astype(np.float64)
    S = np.maximum(np.nan_to_num(S.astype(np.float64), nan=0.0, posinf=0.0), 0.0)
    if h is None:
        h = np.ones(n, dtype=np.float64)
    if p is None:
        p = np.ones(n, dtype=np.float64) * 5.0
    h = np.asarray(h, dtype=np.float64); p = np.asarray(p, dtype=np.float64)

    rev = np.zeros(T, dtype=bool)
    rev[np.clip(review_days, 0, T - 1)] = True

    if init_on_hand is None:
        on_hand = S[0].copy()
    else:
        on_hand = np.maximum(np.asarray(init_on_hand, dtype=np.float64), 0.0)
    pipeline = np.zeros((n, max(lead, 1)), dtype=np.float64)   # arrivals at +1..+lead

    hold_c = np.zeros(n); pen_c = np.zeros(n); ord_c = np.zeros(n)
    sold = np.zeros(n); lost = np.zeros(n); dem_tot = np.zeros(n)
    day_cost = np.zeros((T, n), dtype=np.float32)      # per-day total cost
    day_short = np.zeros((T, n), dtype=np.float32)     # per-day units short
    onhand_trace = np.empty((T, n), dtype=np.float32)
    order_trace = np.zeros((T, n), dtype=np.float32)
    short_days = np.zeros(n)
    n_orders = np.zeros(n)
    stockout_flag = np.zeros(n, dtype=bool)          # any stockout in the window
    # cycle service level: fraction of 28-day periods with no stockout at all
    period_len = 28
    n_per = max(T // period_len, 1)
    per_short = np.zeros((n_per, n), dtype=bool)

    for t in range(T):
        # 1. receive outstanding orders
        if lead > 0:
            on_hand += pipeline[:, 0]
            pipeline[:, :-1] = pipeline[:, 1:]
            pipeline[:, -1] = 0.0
        else:
            on_hand += pipeline[:, 0]
            pipeline[:, 0] = 0.0

        # 2. place an order if this is a review day
        if rev[t]:
            ip = on_hand + pipeline.sum(1)
            q = np.maximum(S[t] - ip, 0.0)
            if min_order > 0:
                q = np.where((q > 0) & (q < min_order), min_order, q)
            place = q > 0
            if place.any():
                ord_c[place] += K
                n_orders[place] += 1
                day_cost[t, place] += K
                if lead > 0:
                    pipeline[place, lead - 1] += q[place]
                else:
                    on_hand[place] += q[place]
                order_trace[t, place] = q[place]

        # 3. satisfy demand (lost sales)
        d = demand[t]
        s = np.minimum(d, on_hand)
        ls = d - s
        on_hand -= s
        sold += s; lost += ls; dem_tot += d
        hc = h * on_hand
        pc = p * ls
        hold_c += hc
        pen_c += pc
        day_cost[t] += hc + pc        # += : the ordering cost for this day was
                                      # already added when the order was placed
        day_short[t] = ls
        hit = ls > 0
        short_days += hit.astype(np.float64)
        stockout_flag |= hit
        if hit.any():
            per_short[min(t // period_len, n_per - 1)][hit] = True
        onhand_trace[t] = on_hand

    total = hold_c + pen_c + ord_c
    assert np.allclose(day_cost.sum(0), total, rtol=1e-4, atol=1e-3), \
        f"daily cost ledger disagrees with window total ({day_cost.sum()} vs {total.sum()})"
    assert np.allclose(day_short.sum(0), lost, rtol=1e-4, atol=1e-3)
    with np.errstate(invalid="ignore", divide="ignore"):
        fill = np.where(dem_tot > 0, sold / np.maximum(dem_tot, 1e-9), 1.0)
    return dict(
        holding=hold_c.astype(np.float64), penalty=pen_c.astype(np.float64),
        ordering=ord_c.astype(np.float64), total=total.astype(np.float64),
        sold=sold, lost=lost, demand=dem_tot, fill_rate=fill.astype(np.float64),
        csl_window=(1.0 - stockout_flag.astype(np.float64)),
        csl_period=(1.0 - per_short.mean(0)).astype(np.float64),
        short_day_share=(short_days / T).astype(np.float64),
        avg_on_hand=(onhand_trace.mean(0)).astype(np.float64),
        n_orders=n_orders,
        onhand_trace=onhand_trace, order_trace=order_trace,
        day_cost=day_cost, day_short=day_short,
    )


# ==========================================================================
# protection-interval target levels
# ==========================================================================
def cumsum_quantiles(Q: np.ndarray, H: int) -> np.ndarray:
    """
    Q : (n_tau, n, H) per-day demand quantiles at the current origin.
    Returns (n_tau, n, H) where entry [..., k] is the quantile-based estimate of
    cumulative demand over the first k+1 days.

    Summing marginal quantiles is an approximation (it ignores the comonotonicity
    gap), and we use the standard correction of scaling by the ratio of the
    square-root-of-horizon to the horizon to avoid over-conservative buffers.
    Both variants are exposed; `conservative=False` is the default.
    """
    cs = np.cumsum(Q, axis=2)
    return cs


def sqrt_rule_correction(Q: np.ndarray, conservative: bool = False) -> np.ndarray:
    if conservative:
        return np.cumsum(Q, axis=2)
    cs = np.cumsum(Q, axis=2)
    H = Q.shape[2]
    k = np.arange(1, H + 1, dtype=np.float64)[None, None, :]
    # blend full comonotonic sum (k) with independence scaling (sqrt k)
    med = cs[:, :, :1]
    spread = cs - med
    return med + spread * np.sqrt(k) / k


def protection_interval_target(Qh: np.ndarray, protection: int,
                               conservative: bool = False) -> np.ndarray:
    """
    Qh : (n_tau, n, H) daily quantiles from the current origin, H >= protection.
    Returns (n_tau, n) order-up-to levels covering `protection` days.
    """
    cs = sqrt_rule_correction(Qh[:, :, :protection], conservative)
    return cs[:, :, -1]


def normal_theory_target(point_fc: np.ndarray, sigma_daily: np.ndarray,
                         protection: int, z: float) -> np.ndarray:
    """
    Classic base-stock level: expected demand over the protection interval plus
    z * sigma * sqrt(protection). This is the policy a forecaster-accuracy-driven
    pipeline produces, and it is exactly what breaks down on lumpy demand.
    """
    mu = np.maximum(point_fc[:, :protection].sum(1), 0.0).astype(np.float64)
    s = np.maximum(np.nan_to_num(sigma_daily.astype(np.float64), nan=0.0), 0.0)
    return (mu + z * s * np.sqrt(protection)).astype(np.float64)


# ==========================================================================
# policies
# ==========================================================================
def build_S_surface(n_days: int, origin_days: list[int],
                    level_by_origin: np.ndarray, n: int) -> np.ndarray:
    """
    Expand per-origin target levels into a (n_days, n) surface.
    level_by_origin : (n_origins, n) -- the S computed at each forecast origin.
    The level set at origin o remains in force until the next origin.
    """
    S = np.zeros((n_days, n), dtype=np.float64)
    bounds = list(origin_days) + [n_days]
    for i, o in enumerate(origin_days):
        end = bounds[i + 1]
        S[o:end] = level_by_origin[i]
    if origin_days[0] > 0:
        S[:origin_days[0]] = level_by_origin[0]      # carry the first level backwards
    return np.maximum(S, 0.0)


def policy_base_stock_normal(point_fc_by_origin, sigma_daily, protection, z,
                             origin_days, n_days, n):
    """P1: point forecast + normal-theory safety stock (the accuracy-driven default)."""
    levels = np.stack([normal_theory_target(f, sigma_daily, protection, z)
                       for f in point_fc_by_origin], axis=0)
    return build_S_surface(n_days, origin_days, levels, n)


def policy_quantile(Q_by_origin, protection, tau_ix, origin_days, n_days, n,
                    conservative=False):
    """P2: order-up-to level taken directly at the critical-ratio quantile."""
    levels = np.stack([protection_interval_target(Q, protection, conservative)[tau_ix]
                       for Q in Q_by_origin], axis=0)   # (n_orig, n)
    return build_S_surface(n_days, origin_days, levels, n)


def policy_sS(Q_by_origin, protection, tau_ix, origin_days, n_days, n,
              s_frac: float = 0.5):
    """P3: (s, S) with a fixed order cost. S from the quantile surface,
    s = s_frac * S, reviewed daily."""
    S_lv = np.stack([protection_interval_target(Q, protection)[tau_ix]
                     for Q in Q_by_origin], axis=0)
    S_surf = build_S_surface(n_days, origin_days, S_lv, n)
    s_surf = s_frac * S_surf
    return S_surf, s_surf


def simulate_sS(demand, S_surf, s_surf, lead, h, p, K, review_days=None):
    """
    (s, S) policy: order up to S whenever the inventory position falls to or
    below s. `review_days` selects the days on which the trigger is evaluated
    (default: every day).
    """
    T, n = demand.shape
    rev = np.ones(T, dtype=bool)
    if review_days is not None:
        rev = np.zeros(T, dtype=bool)
        rev[np.clip(np.asarray(review_days, dtype=np.int64), 0, T - 1)] = True
    demand = demand.astype(np.float64)
    S = np.maximum(np.nan_to_num(S_surf, nan=0.0), 0.0)
    s = np.maximum(np.nan_to_num(s_surf, nan=0.0), 0.0)
    on_hand = S[0].copy()
    pipeline = np.zeros((n, max(lead, 1)), dtype=np.float64)
    hold_c = np.zeros(n); pen_c = np.zeros(n); ord_c = np.zeros(n)
    sold = np.zeros(n); lost = np.zeros(n); dem = np.zeros(n)
    short_days = np.zeros(n); so_flag = np.zeros(n, dtype=bool)
    onhand_trace = np.empty((T, n), dtype=np.float32)
    day_cost = np.zeros((T, n), dtype=np.float32)
    day_short = np.zeros((T, n), dtype=np.float32)
    n_orders = np.zeros(n)
    period_len = 28
    n_per = max(T // period_len, 1)
    per_short = np.zeros((n_per, n), dtype=bool)
    for t in range(T):
        on_hand += pipeline[:, 0]
        if lead > 1:
            pipeline[:, :-1] = pipeline[:, 1:]
        pipeline[:, -1] = 0.0
        ip = on_hand + pipeline.sum(1)
        q = (np.where(ip <= s[t], np.maximum(S[t] - ip, 0.0), 0.0)
             if rev[t] else np.zeros(n))
        place = q > 0
        if place.any():
            ord_c[place] += K
            n_orders[place] += 1
            pipeline[place, lead - 1] += q[place]
            day_cost[t, place] += K
        d = demand[t]
        sl = np.minimum(d, on_hand)
        ls = d - sl
        on_hand -= sl
        sold += sl; lost += ls; dem += d
        hc = h * on_hand; pc = p * ls
        hold_c += hc; pen_c += pc
        day_cost[t] += hc + pc
        day_short[t] = ls
        hit = ls > 0
        short_days += hit; so_flag |= hit
        if hit.any():
            per_short[min(t // period_len, n_per - 1)][hit] = True
        onhand_trace[t] = on_hand
    total = hold_c + pen_c + ord_c
    assert np.allclose(day_cost.sum(0), total, rtol=1e-4, atol=1e-3), \
        f"daily cost ledger disagrees with window total ({day_cost.sum()} vs {total.sum()})"
    assert np.allclose(day_short.sum(0), lost, rtol=1e-4, atol=1e-3)
    fill = np.where(dem > 0, sold / np.maximum(dem, 1e-9), 1.0)
    return dict(holding=hold_c, penalty=pen_c, ordering=ord_c, total=total,
                sold=sold, lost=lost, demand=dem, fill_rate=fill,
                csl_window=1.0 - so_flag.astype(np.float64),
                csl_period=1.0 - per_short.mean(0),
                short_day_share=short_days / T,
                avg_on_hand=onhand_trace.mean(0),
                n_orders=n_orders, onhand_trace=onhand_trace,
                day_cost=day_cost, day_short=day_short)


def policy_ddmrp(point_fc_by_origin, protection, origin_days, n_days, n,
                 lt_days: int, buffer_frac: float = 0.5,
                 green_frac: float = 0.35, yellow_frac: float = 0.25,
                 review: int = 7):
    """
    P4: a demand-driven MRP style buffered policy (Orlicky's material logic,
    as popularised by Ptak & Smith).

      ADG  = average daily demand over the protection interval
      buf  = buffer_frac * ADG * (lt_days + review)          total buffer
      red  = yellow_frac * buf ; yellow = green_frac * buf ; green = buf-red-yellow
      top-of-green = ADG*(lt+review) + green + yellow + red  = order-up-to S
      top-of-red   = ADG*lt + red                            = reorder point s

    The buffer sits *on top of* the cycle-stock requirement ADG*(L+R); sizing the
    buffer as the whole order-up-to level (a common shortcut) collapses S and s
    together and degenerates into ordering on nearly every demand event.

    Deliberately robust: it uses only the *level* of the forecast, never its
    distribution, so it isolates how much of the benefit of distributional
    forecasting survives a simple buffered rule.
    """
    adg = np.stack([np.maximum(f[:, :protection].mean(1), 0.0)
                    for f in point_fc_by_origin], axis=0)     # (n_orig, n)
    buf = buffer_frac * adg * (lt_days + review)
    red = yellow_frac * buf
    yellow = green_frac * buf
    green = np.maximum(buf - red - yellow, 0.0)
    S_lv = adg * (lt_days + review) + green + yellow + red
    s_lv = adg * lt_days + red
    S_surf = build_S_surface(n_days, origin_days, S_lv, n)
    s_surf = build_S_surface(n_days, origin_days, s_lv, n)
    return S_surf, s_surf


# ==========================================================================
# aggregate statistics
# ==========================================================================
def summarise(res: dict, groups: np.ndarray | None = None) -> dict:
    """Window-level, series-averaged KPIs."""
    dem = res["demand"].sum()
    out = dict(
        total_cost=float(res["total"].sum()),
        holding_cost=float(res["holding"].sum()),
        penalty_cost=float(res["penalty"].sum()),
        ordering_cost=float(res["ordering"].sum()),
        mean_cost_per_series=float(res["total"].mean()),
        fill_rate=float(res["sold"].sum() / max(dem, 1e-9)),
        csl=float(res["csl_period"].mean()),
        short_day_share=float(res["short_day_share"].mean()),
        lost_units=float(res["lost"].sum()),
        demand_units=float(dem),
        avg_on_hand=float(res["avg_on_hand"].mean()),
        cost_per_unit_demand=float(res["total"].sum() / max(dem, 1e-9)),
    )
    return out
