"""Rule-based ball-position estimates for stretches with missing tracking.

Observed ATD coordinates are never changed. Short gaps use a smooth path between the
observations on either side; a nearby tracked player can pull that path toward a likely
controlled touch. LONG BALL tags suppress that pull, since the ball is more likely to be in
flight. Estimates carry a confidence that downstream possession evidence can use as a weight.

This is deliberately a modest imputer, not a claim that long unobserved trajectories can be
recovered exactly. Long gaps without player support remain low-confidence estimates.
"""
import numpy as np


def _runs(mask):
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    changes = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _long_ball_mask(tags, n):
    mask = np.zeros(n, bool)
    for tag in tags or ():
        if (tag.get("code") or "").strip().upper() != "LONG BALL":
            continue
        a = max(0, int(tag.get("start_frame", 0)))
        b = min(n, int(tag.get("end_frame", a)))
        if b > a:
            mask[a:b] = True
    return mask


def _segments(n, blocks=None, period=None):
    """Half-open spans; never interpolate across half-time or an explicit frame block."""
    if blocks:
        return [(max(0, int(a)), min(n, int(b) + 1)) for _, a, b in blocks if b >= a]
    if period is not None and len(period) == n:
        return [(int(ix[0]), int(ix[-1]) + 1)
                for p in np.unique(period) if len(ix := np.flatnonzero(period == p))]
    return [(0, n)]


def predict(ball, players, tags=None, blocks=None, period=None, fps=25):
    """Return ``(estimated_xy, confidence)`` without modifying observed ball coordinates.

    ``ball`` is (frames, 2), ``players`` is (frames, slots, 2), both in the same pitch units.
    Confidence is 1 on observed frames, falls with gap length, and increases modestly when a
  visible player is close to the interpolated trajectory. One-sided extrapolation is limited
    to half a second and is always lower confidence than a two-sided estimate.
    """
    ball = np.asarray(ball, np.float32)
    players = np.asarray(players, np.float32)
    n = len(ball)
    estimate = ball.copy()
    observed = np.isfinite(ball).all(axis=1)
    confidence = observed.astype(np.float32)
    flight = _long_ball_mask(tags, n)

    for lo, hi in _segments(n, blocks, period):
        if hi <= lo:
            continue
        missing_runs = _runs(~observed[lo:hi])
        for a0, b0 in missing_runs:
            a, b = lo + a0, lo + b0
            length = b - a
            if length <= 0:
                continue
            left, right = a - 1, b
            has_left = left >= lo and observed[left]
            has_right = right < hi and observed[right]

            if has_left and has_right:
                frac = (np.arange(a, b, dtype=np.float32) - left) / float(right - left)
                path = ball[left][None, :] + frac[:, None] * (ball[right] - ball[left])[None, :]
                # Interpolation is most useful for short gaps; uncertainty grows quickly after
                # a few seconds, even when both endpoints are known.
                base_conf = 0.82 * np.exp(-length / float(4 * fps))
                conf = np.full(length, max(0.12, base_conf), np.float32)

                # For a likely controlled touch, pull only gently toward a nearby player. During
                # a tagged long ball retain the free trajectory instead of snapping to a player.
                pp = players[a:b]
                d2 = ((pp - path[:, None, :]) ** 2).sum(axis=2)
                d2[~np.isfinite(d2)] = np.inf
                nearest_slot = np.argmin(d2, axis=1)
                nearest_d = np.sqrt(np.min(d2, axis=1))
                snap = (nearest_d <= 2.25) & ~flight[a:b]
                if np.any(snap):
                    rows = np.flatnonzero(snap)
                    carrier_xy = pp[rows, nearest_slot[rows]]
                    estimate[a + rows] = 0.35 * path[rows] + 0.65 * carrier_xy
                    conf[rows] = np.maximum(conf[rows], 0.48 * np.exp(-length / float(12 * fps)))
                gap_estimate = estimate[a:b]
                gap_estimate[~snap] = path[~snap]
                estimate[a:b] = gap_estimate
                confidence[a:b] = conf

            elif has_left or has_right:
                # A bounded constant-velocity extrapolation avoids inventing a long path past
                # the first/last ball observation in a half.
                anchor = left if has_left else right
                direction = 1.0 if has_left else -1.0
                prior = anchor - 1 if has_left else anchor + 1
                if lo <= prior < hi and observed[prior]:
                    velocity = (ball[anchor] - ball[prior]) * fps * direction
                    speed = float(np.linalg.norm(velocity))
                    if speed > 25.0:
                        velocity *= 25.0 / speed
                else:
                    velocity = np.zeros(2, np.float32)
                frames = np.arange(a, b, dtype=np.float32)
                dt = (frames - anchor) / float(fps)
                use = np.abs(dt) <= 0.5
                if np.any(use):
                    pos = ball[anchor][None, :] + dt[use, None] * velocity[None, :]
                    dest = frames[use].astype(int)
                    estimate[dest] = pos
                    confidence[dest] = (0.30 * np.exp(-np.abs(dt[use]) / 0.35)).astype(np.float32)

    return estimate, confidence
