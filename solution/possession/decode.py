"""Minimum-duration HMM over the states H, A, D, decoded with Viterbi.

Each state is expanded into a chain of `min_len` sub-states, so a state can only
be left after it has lasted at least min_len frames: a possession of 3 frames is
impossible by construction, which is what the segment metric (spellF1) rewards.
Leaving a state costs a fixed log-penalty.
"""
import numpy as np

from .emissions import H, A, D, STATES

DECODE = {
    "min_len": {H: 25, A: 25, D: 50},     # frames: 1 s possession, 2 s stoppage
    "switch_cost": {                      # log-penalty for (from, to)
        (H, A): 6.0, (A, H): 6.0,
        (H, D): 6.0, (A, D): 6.0,
        (D, H): 6.0, (D, A): 6.0,
    },
}


def _build(cfg):
    """Sub-state layout and the list of allowed predecessor edges."""
    offs, owner = {}, []
    for s in (H, A, D):
        offs[s] = len(owner)
        owner += [s] * cfg["min_len"][s]
    owner = np.array(owner)
    K = len(owner)
    # predecessors as dense matrix of log-transition scores (-inf = not allowed)
    T = np.full((K, K), -np.inf, np.float32)
    for s in (H, A, D):
        o, m = offs[s], cfg["min_len"][s]
        for k in range(m - 1):
            T[o + k, o + k + 1] = 0.0          # forced progress through the chain
        T[o + m - 1, o + m - 1] = 0.0          # stay once the minimum is reached
        for t in (H, A, D):
            if t != s:
                T[o + m - 1, offs[t]] = -cfg["switch_cost"][(s, t)]
    return owner, offs, T


def viterbi(ll, cfg=DECODE, start=None, team_switch_guard=None):
    """ll: (n, 3) per-frame scores. Returns state index per frame."""
    owner, offs, T = _build(cfg)
    K, n = len(owner), len(ll)
    em = ll[:, owner]                                   # (n, K)
    delta = np.full(K, -np.inf, np.float32)
    starts = [start] if start is not None else [H, A, D]
    for s in starts:
        delta[offs[s]] = 0.0
    delta = delta + em[0]
    back = np.empty((n, K), np.int16)
    back[0] = np.arange(K)
    for t in range(1, n):
        cand = delta[:, None] + T                       # (K_from, K_to)
        if team_switch_guard is not None:
            guard = team_switch_guard[t]
            cand[offs[H] + cfg["min_len"][H] - 1, offs[A]] -= guard[0]
            cand[offs[A] + cfg["min_len"][A] - 1, offs[H]] -= guard[1]
        back[t] = np.argmax(cand, 0)
        delta = cand[back[t], np.arange(K)] + em[t]
    path = np.empty(n, np.int64)
    k = int(np.argmax(delta))
    for t in range(n - 1, -1, -1):
        path[t] = k
        k = back[t, k]
    return owner[path]


def decode_match(ll, blocks, n, cfg=DECODE, team_switch_guard=None):
    """Decode each graded half independently; X outside the graded frames."""
    labels = np.full(n, "X", dtype="<U1")
    for _, a, b in blocks:
        guard = None if team_switch_guard is None else team_switch_guard[a:b + 1]
        labels[a:b + 1] = STATES[viterbi(ll[a:b + 1], cfg, team_switch_guard=guard)]
    return labels


def confirm_team_changes(labels, control_home, control_away, params, blocks, context=None):
    """Treat an unconfirmed H/A challenger as a latent contest and retain the incumbent.

    A change is emitted only after the challenger has sufficient team-level control support
    for a trailing window. D transitions remain immediate and restart into H/A is unguarded.
    Since support is pooled by team, player-to-player handoffs do not reset confirmation.
    """
    window = int(params.get("team_change_confirm_frames", 0))
    if window <= 1:
        return labels
    fraction = float(params.get("team_change_confirm_fraction", 0.7))
    strength_floor = float(params.get("team_change_min_strength", 0.15))
    margin = float(params.get("team_change_min_margin", 0.05))
    contextual = bool(params.get("team_change_contextual", 0))
    flight_extra = int(params.get("team_change_flight_extra_frames", 10))
    contest_extra = int(params.get("team_change_contested_extra_frames", 10))
    action_bonus = int(params.get("team_change_clear_action_frames", 7))
    action_threshold = float(params.get("team_change_acquisition_threshold", 0.25))
    contest_margin = float(params.get("team_change_contested_margin", 0.15))
    flight_speed = float(params.get("flight_speed", 10.0))
    out = labels.copy()
    for _, first, last in blocks:
        incumbent = out[first]
        pending = None
        support = []
        for frame in range(first + 1, last + 1):
            candidate = labels[frame]
            if candidate not in (STATES[H], STATES[A]):
                out[frame] = candidate
                pending, support = None, []
                incumbent = candidate
                continue
            if incumbent not in (STATES[H], STATES[A]) or candidate == incumbent:
                incumbent = candidate
                out[frame] = candidate
                pending, support = None, []
                continue
            if candidate != pending:
                pending, support = candidate, []
            if candidate == STATES[H]:
                new_score, old_score = control_home[frame], control_away[frame]
                new_action = (context.get("acquisition_home", np.zeros(len(labels)))[frame]
                              if context else 0.0)
            else:
                new_score, old_score = control_away[frame], control_home[frame]
                new_action = (context.get("acquisition_away", np.zeros(len(labels)))[frame]
                              if context else 0.0)
            action_support = new_action >= action_threshold and new_score >= strength_floor
            support.append(bool((new_score >= strength_floor and new_score - old_score >= margin)
                                or action_support))
            required = window
            if contextual and context:
                speed = context.get("ball_speed", np.zeros(len(labels)))[frame]
                long_ball = context.get("long_ball", np.zeros(len(labels), bool))[frame]
                if long_ball or speed > flight_speed:
                    required += flight_extra
                both_near = min(control_home[frame], control_away[frame]) >= strength_floor
                if both_near and abs(control_home[frame] - control_away[frame]) <= contest_margin:
                    required += contest_extra
                if action_support:
                    required = max(5, required - action_bonus)
            while len(support) > required:
                support.pop(0)
            if len(support) == required and np.mean(support) >= fraction:
                incumbent = candidate
                out[frame] = candidate
                pending, support = None, []
            else:
                out[frame] = incumbent
    return out
