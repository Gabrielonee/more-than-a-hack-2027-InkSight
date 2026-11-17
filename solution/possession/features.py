"""Per-frame features computed from tracking and rule-based ball-position estimates.

All distances in metres, speeds in m/s. Raw ball visibility remains separate from estimate
confidence so downstream rules can distinguish observations from reconstructed positions.
"""
import numpy as np
import pandas as pd

from .data import FPS, PITCH, HOME, AWAY
from . import ball_position


def runs(mask):
    """[(start, end_exclusive)] of consecutive True values."""
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _central_speed(xy, h=2):
    """Speed from a centred difference over +-h frames; NaN where either end is missing."""
    velocity = _central_velocity(xy, h)
    return np.sqrt((velocity ** 2).sum(-1))


def _central_velocity(xy, h=2):
    """Vector velocity from a centred difference over +-h frames."""
    v = np.full(xy.shape, np.nan, np.float32)
    dxy = xy[2 * h:] - xy[:-2 * h]
    v[h:-h] = dxy * FPS / (2 * h)
    return v


def rolling(x, window, how="mean", min_periods=1):
    s = pd.Series(x)
    r = s.rolling(window, center=True, min_periods=min_periods)
    return getattr(r, how)().to_numpy(np.float32)


def compute(atd, tags=None, blocks=None):
    observed_ball, players, team = atd["ball"], atd["players"], atd["team"]
    L, W = PITCH
    f = {}
    present = np.isfinite(observed_ball).all(axis=1)
    period = atd.get("period")
    predicted_ball, predicted_confidence = ball_position.predict(
        observed_ball, players, tags=tags, blocks=blocks, period=period, fps=FPS)
    ball = observed_ball.copy()
    missing = ~present
    use_prediction = missing & np.isfinite(predicted_ball).all(axis=1)
    ball[use_prediction] = predicted_ball[use_prediction]
    location_confidence = np.where(present, 1.0, predicted_confidence).astype(np.float32)
    f["ball_present"] = present
    f["ball_position_confidence"] = location_confidence
    f["ball_for_geometry"] = ball
    f["team_of_track"] = team
    long_ball = np.zeros(len(ball), bool)
    long_ball_release = np.zeros(len(ball), np.float32)
    for tag in tags or ():
        if (tag.get("code") or "").strip().upper() == "LONG BALL":
            a = max(0, int(tag.get("start_frame", 0)))
            b = min(len(ball), int(tag.get("end_frame", a)))
            if b > a:
                long_ball[a:b] = True
                end = min(len(ball), b + FPS)
                if end > b:
                    long_ball_release[b:end] = np.linspace(1.0, 0.0, end - b, endpoint=False)
    f["long_ball"] = long_ball
    f["long_ball_release"] = long_ball_release

    # ball speed, robust to single-frame jumps of the automatic tracker
    bs = _central_speed(ball)
    f["ball_speed"] = rolling(bs, 5, "median")

    # Distances and relative motion support a per-player control score downstream.
    d = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))          # (N, 46)
    player_v = _central_velocity(players)
    ball_v = _central_velocity(ball)
    relative_speed = np.sqrt(((player_v - ball_v[:, None, :]) ** 2).sum(-1))
    distance_rate = np.full(d.shape, np.nan, np.float32)
    distance_rate[1:-1] = (d[2:] - d[:-2]) * FPS / 2.0
    approaching = np.maximum(-distance_rate, 0.0)
    nearby = np.where(np.isfinite(d), (d < 2.5).astype(np.float32), np.nan)
    persistence = np.stack([rolling(nearby[:, j], FPS, "mean")
                            for j in range(nearby.shape[1])], axis=1)
    ball_accel = np.full(len(ball), np.nan, np.float32)
    dv = ball_v[4:] - ball_v[:-4]
    ball_accel[2:-2] = np.sqrt((dv ** 2).sum(-1)) * FPS / 4.0
    ball_accel = rolling(ball_accel, 5, "median")
    speed_rate = np.full(len(ball), np.nan, np.float32)
    speed_rate[2:-2] = (f["ball_speed"][4:] - f["ball_speed"][:-4]) * FPS / 4.0
    ball_deceleration = rolling(np.maximum(-speed_rate, 0.0), 5, "median")
    f["player_ball_distance"] = d
    f["player_relative_speed"] = relative_speed
    f["player_approach_speed"] = approaching
    f["player_nearby_persistence"] = persistence
    f["ball_acceleration"] = ball_accel
    f["ball_deceleration"] = ball_deceleration

    # Nearest distance by team remains useful to restart and gap features.
    with np.errstate(all="ignore"):
        f["d_home"] = np.where(np.isnan(d[:, team == HOME]).all(1), np.nan,
                               np.nanmin(np.where(np.isnan(d[:, team == HOME]), np.inf, d[:, team == HOME]), 1))
        f["d_away"] = np.where(np.isnan(d[:, team == AWAY]).all(1), np.nan,
                               np.nanmin(np.where(np.isnan(d[:, team == AWAY]), np.inf, d[:, team == AWAY]), 1))

    # how far the ball is outside the pitch (<=0 inside) and from the nearest line
    x, y = ball[:, 0], ball[:, 1]
    out = np.maximum.reduce([-x, x - L, -y, y - W])
    f["ball_outside"] = out.astype(np.float32)
    f["ball_to_line"] = np.abs(out).astype(np.float32)

    # collective player speed: players walk during stoppages
    ps = _central_speed(players)                                        # (N, 46)
    ps = np.where(ps > 12, np.nan, ps)                                  # identity switches
    with np.errstate(all="ignore"):
        mean_speed = np.nanmean(ps, 1)
    f["players_speed"] = rolling(mean_speed, 2 * FPS, "mean")
    f["n_players"] = (~np.isnan(players[:, :, 0])).sum(1)

    # ball stillness over 1 s (placed ball before a restart)
    f["ball_still"] = rolling((f["ball_speed"] < 1.0).astype(np.float32), FPS, "mean")
    return f
