"""Calibrate the hand-set parameters on the Metrica open games (never on BAR-ATH).

Objective: mean spellF1 (the official metric code) over the two sample games, with the
ball removed in long blocks to resemble the automatic tracking. Random search, seeded.
"""
import copy
import json
import os
import sys

import numpy as np

from . import features, emissions, decode, metrica, gaps, deadball, setpieces
from .data import ROOT, CACHE_DIR

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

SEARCH = {  # (low, high) sampled uniformly
    "k_team": (0.1, 1.5),
    "control_decay": (0.5, 3.0),
    "flight_speed": (6.0, 16.0),
    "controller_relative_speed": (0.8, 6.0),
    "controller_action_weight": (0.0, 0.5),
    "controller_acquisition_weight": (0.0, 0.5),
    "slow_center": (0.8, 1.8),
    "k_slow": (0.05, 0.8),
    "k_still": (0.0, 0.8),
    "d_bias": (-0.6, 0.1),
    "k_dead": (0.0, 0.8),       # k_gap not searched: the dead-ball model already sees the gaps
    "k_sp_dead": (0.0, 1.0),    # set-piece anchors, calibrated with synthetic tags (metrica.py)
    "k_sp_team": (0.0, 1.5),
    "sp_back_scale": (0.5, 1.5),
}
DECODE_SEARCH = {"switch": (2.0, 15.0), "switch_team": (0.5, 10.0),
                 "min_dead": (10, 150)}
MIN_POS_CANDIDATES = (10, 15, 20)


def prepare(games=(1, 2), seed=0, ball_mode="atd_like", gap_model=False, dead_model=True):
    """[(G, f, gt, gap_logit, dead_logit, synthetic_tags)] per game.

    ball_mode: "atd_like" (default) fills the native Metrica gaps and cuts long random gaps, so a
    missing ball is no longer a proxy for a dead ball (as in the automatic tracking);
    "legacy" is the v1 protocol (native gaps + extra random gaps, 55% missing).
    gap_model: the gap classifier for each game is trained on the OTHER game (leave-one-game-out),
    so the calibration never sees a classifier that was fitted on the game it is scored on.
    dead_model: same leave-one-game-out scheme for the dead-ball classifier (deadball.py).
    """
    rng = np.random.default_rng(seed)
    out = []
    for g in games:
        G = metrica.load_game(g)
        gt = metrica.labels_from_events(G)
        if ball_mode == "atd_like":
            b = metrica.atd_like_ball(G["ball"], G["period"], rng)
        else:
            b = metrica.drop_ball_like_atd(G["ball"], rng, target=0.55)
        G = dict(G, ball=b)
        out.append((G, features.compute(G), gt))
    gl = [np.zeros(G["n_frames"], np.float32) for G, _, _ in out]
    if gap_model and len(out) > 1:
        sets = gaps.training_set(out)
        for i, (G, f, gt) in enumerate(out):
            X = np.concatenate([s[0] for j, s in enumerate(sets) if j != i])
            y = np.concatenate([s[1] for j, s in enumerate(sets) if j != i])
            gl[i] = gaps.gap_logit(f, G["ball"], blocks_of(gt), gaps.fit(X, y))
    dl = [np.zeros(G["n_frames"], np.float32) for G, _, _ in out]
    if dead_model and len(out) > 1:
        sets = deadball.training_set(out)
        for i, (G, f, gt) in enumerate(out):
            X = np.concatenate([s[0] for j, s in enumerate(sets) if j != i])
            y = np.concatenate([s[1] for j, s in enumerate(sets) if j != i])
            dl[i] = deadball.dead_logit(f, G, deadball.fit(X, y))
    tags = [metrica.synthetic_tags(G, rng) for G, _, _ in out]
    return [(G, f, gt, l, d, t) for (G, f, gt), l, d, t in zip(out, gl, dl, tags)]


def fit_gap_model(prepared):
    """Final gap classifier, fitted on ALL the prepared Metrica games (for the challenge game)."""
    sets = gaps.training_set(prepared)
    return gaps.fit(np.concatenate([s[0] for s in sets]), np.concatenate([s[1] for s in sets]))


def blocks_of(gt):
    codes = pm.label_codes(gt)
    return [(0, a, b - 1) for a, b in pm.valid_runs(codes)]


def fit_dead_model(prepared):
    """Final dead-ball classifier, fitted on ALL the prepared Metrica games."""
    sets = deadball.training_set(prepared)
    return deadball.fit(np.concatenate([s[0] for s in sets]), np.concatenate([s[1] for s in sets]))


def score(prepared, params, cfg):
    s = []
    for G, f, gt, gl, dl, tags in prepared:
        ll = (emissions.tracking_scores(f, params) + emissions.gap_scores(gl, params)
              + emissions.dead_scores(dl, params)
              + setpieces.setpiece_scores(tags, f, G["ball"], G["n_frames"], params)[0])
        guard = emissions.team_switch_guard(f, params)
        pred = decode.decode_match(ll, blocks_of(gt), G["n_frames"], cfg,
                                   team_switch_guard=guard)
        pred = decode.confirm_team_changes(pred, f.get("control_home", f["d_home"]),
                                           f.get("control_away", f["d_away"]), params,
                                           blocks_of(gt), context=f)
        gtc, prc = pm.label_codes(gt), pm.label_codes(pred)
        s.append(pm.spell_scores(gtc, prc, pm.valid_runs(gtc))["mean_f1"])
    return float(np.mean(s)), s


def make_cfg(sw, mp, md, switch_team=None):
    cfg = copy.deepcopy(decode.DECODE)
    cfg["min_len"] = {emissions.H: int(mp), emissions.A: int(mp), emissions.D: int(md)}
    cfg["switch_cost"] = {k: float(sw) for k in cfg["switch_cost"]}
    if switch_team is not None:
        cfg["switch_cost"][(emissions.H, emissions.A)] = float(switch_team)
        cfg["switch_cost"][(emissions.A, emissions.H)] = float(switch_team)
    return cfg


def sample(rng):
    p = dict(emissions.PARAMS)
    for k, (lo, hi) in SEARCH.items():
        p[k] = float(rng.uniform(lo, hi))
    d = {k: float(rng.uniform(*v)) for k, v in DECODE_SEARCH.items()}
    d["min_pos"] = int(rng.choice(MIN_POS_CANDIDATES))
    return p, d


def random_search(n_trials, seed, out_path=None, prepared=None):
    prepared = prepared or prepare()
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_trials):
        p, d = sample(rng)
        m, per = score(prepared, p, make_cfg(d["switch"], d["min_pos"], d["min_dead"],
                                             d["switch_team"]))
        results.append({"spell": m, "per_game": per, "params": p, "decode": d})
        if out_path:
            with open(out_path, "a") as fh:
                fh.write(json.dumps(results[-1]) + "\n")
    return results


if __name__ == "__main__":
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    random_search(n, seed, os.path.join(CACHE_DIR, "calibration.jsonl"))


# --- parallel search with checkpoint/resume --------------------------------------------------
# Every finished trial is appended to a JSONL log (on Google Drive when running on Colab), so a
# runtime reset loses at most the trials in flight; rerunning the same call resumes from the log.
# Trial i of seed s always samples the same configuration (rng seeded with [s, i]).

_PREPARED = None


def _init_worker():
    global _PREPARED
    _PREPARED = prepare()          # once per process, not once per trial


def _eval_trial(task):
    seed, i, *min_pos = task
    p, d = sample(np.random.default_rng([seed, i]))
    if min_pos:
        d["min_pos"] = int(min_pos[0])
    m, per = score(_PREPARED, p, make_cfg(d["switch"], d["min_pos"], d["min_dead"],
                                          d["switch_team"]))
    return {"seed": seed, "trial": i, "spell": m, "per_game": per, "params": p, "decode": d}


def load_log(path):
    """Records in a JSONL log; a truncated last line (runtime killed mid-write) is skipped."""
    out = []
    if path and os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def parallel_search(n_trials, n_jobs, seed, log_path=None,
                    min_pos_values=MIN_POS_CANDIDATES):
    """Random search over processes. With log_path: resumes from the log and returns ALL the
    records in it (keep one log per code version: old records are scored with old code)."""
    from multiprocessing import Pool
    done = load_log(log_path)
    sweep = tuple(int(v) for v in min_pos_values) if min_pos_values else (None,)
    have = {(r.get("trial"), r.get("decode", {}).get("min_pos"))
            for r in done if r.get("seed") == seed and "trial" in r}
    todo = [(seed, i, mp) for i in range(n_trials) for mp in sweep
            if (i, mp) not in have]
    print(f"log: {len(done)} configurazioni gia' fatte, {len(todo)} da fare")
    if todo:
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with Pool(max(1, min(n_jobs, len(todo))), initializer=_init_worker) as pool:
            for k, r in enumerate(pool.imap_unordered(_eval_trial, todo), 1):
                done.append(r)
                if log_path:
                    with open(log_path, "a") as fh:
                        fh.write(json.dumps(r) + "\n")
                        fh.flush()
                        os.fsync(fh.fileno())
                if k % 10 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)}  migliore finora (max-min): {max(min(x['per_game']) for x in done):.3f}",
                          flush=True)
    return done


def best_of(results):
    """Robust choice; break equal worst-game scores by mean score."""
    return max(results, key=lambda r: (min(r["per_game"]), r["spell"]))


def best_by_min_pos(results):
    """Best robust config for each tested minimum live-possession duration."""
    return {mp: max((r for r in results if int(r["decode"]["min_pos"]) == mp),
                    key=lambda r: (min(r["per_game"]), r["spell"]))
            for mp in sorted({int(r["decode"]["min_pos"]) for r in results})}


def controller_stability_sweep(params, cfg, seeds=(0, 1, 2, 3),
                               windows=(0, 10, 15, 20)):
    """Compare same-player control persistence across fixed masks/tags from each seed.

    Select by worst per-game spellF1, then mean. This ablation keeps every learned/decoded
    parameter fixed and only changes the trailing control window.
    """
    records = []
    for seed in seeds:
        prepared = prepare(seed=seed)
        for window in windows:
            p = dict(params, controller_stability_frames=int(window))
            mean, per_game = score(prepared, p, cfg)
            records.append({"seed": int(seed), "window": int(window),
                            "spell": mean, "per_game": per_game})
    summary = {}
    for window in windows:
        rows = [r for r in records if r["window"] == int(window)]
        scores = [v for row in rows for v in row["per_game"]]
        summary[int(window)] = {
            "mean": float(np.mean(scores)), "worst": float(np.min(scores)),
            "per_seed_mean": float(np.mean([row["spell"] for row in rows])),
        }
    selected = max(summary, key=lambda w: (summary[w]["worst"], summary[w]["mean"]))
    return {"records": records, "summary": summary, "selected_window": int(selected)}


def team_dwell_sweep(params, cfg, seeds=(0, 1, 2, 3),
                     min_pos_values=(20, 30, 40, 50)):
    """Test how long a challenger team must persist before a possession change is allowed.

    The HMM's minimum H/A duration acts as a team-level dwell time, independent of player
    identity: a handoff between teammates remains the same possession state.
    """
    records = []
    for seed in seeds:
        prepared = prepare(seed=seed)
        for min_pos in min_pos_values:
            c = copy.deepcopy(cfg)
            c["min_len"][emissions.H] = int(min_pos)
            c["min_len"][emissions.A] = int(min_pos)
            mean, per_game = score(prepared, params, c)
            records.append({"seed": int(seed), "min_pos": int(min_pos),
                            "spell": mean, "per_game": per_game})
    summary = {}
    for min_pos in min_pos_values:
        rows = [r for r in records if r["min_pos"] == int(min_pos)]
        scores = [v for row in rows for v in row["per_game"]]
        summary[int(min_pos)] = {
            "mean": float(np.mean(scores)), "worst": float(np.min(scores)),
        }
    selected = max(summary, key=lambda m: (summary[m]["worst"], summary[m]["mean"]))
    return {"records": records, "summary": summary, "selected_min_pos": int(selected)}


def team_control_sweep(params, cfg, seeds=(0, 1, 2, 3), variants=None):
    """Compare temporal smoothing before and after pooling candidate players by team."""
    variants = variants or {
        "player15": {"controller_stability_frames": 15, "team_control_stability_frames": 0},
        "team15": {"controller_stability_frames": 0, "team_control_stability_frames": 15},
        "team30": {"controller_stability_frames": 0, "team_control_stability_frames": 30},
        "player15_team15": {"controller_stability_frames": 15,
                             "team_control_stability_frames": 15},
    }
    records = []
    for seed in seeds:
        prepared = prepare(seed=seed)
        for name, update in variants.items():
            p = dict(params)
            p.update(update)
            mean, per_game = score(prepared, p, cfg)
            records.append({"seed": int(seed), "variant": name, "spell": mean,
                            "per_game": per_game})
    summary = {}
    for name in variants:
        rows = [r for r in records if r["variant"] == name]
        scores = [v for row in rows for v in row["per_game"]]
        summary[name] = {"mean": float(np.mean(scores)), "worst": float(np.min(scores))}
    selected = max(summary, key=lambda name: (summary[name]["worst"], summary[name]["mean"]))
    return {"records": records, "summary": summary, "selected_variant": selected}


def team_switch_guard_sweep(params, cfg, seeds=(0, 1, 2, 3), variants=None):
    """Test challenger persistence only on H↔A transitions, leaving D rules untouched."""
    variants = variants or {
        "off": {"team_switch_support_window": 0, "team_switch_support_cost": 0.0},
        "w15_c2": {"team_switch_support_window": 15, "team_switch_support_cost": 2.0},
        "w25_c2": {"team_switch_support_window": 25, "team_switch_support_cost": 2.0},
        "w25_c4": {"team_switch_support_window": 25, "team_switch_support_cost": 4.0},
        "w40_c4": {"team_switch_support_window": 40, "team_switch_support_cost": 4.0},
        "w40_c6": {"team_switch_support_window": 40, "team_switch_support_cost": 6.0},
    }
    records = []
    for seed in seeds:
        prepared = prepare(seed=seed)
        for name, update in variants.items():
            p = dict(params)
            p.update(update)
            mean, per_game = score(prepared, p, cfg)
            records.append({"seed": int(seed), "variant": name, "spell": mean,
                            "per_game": per_game})
    summary = {}
    for name in variants:
        rows = [r for r in records if r["variant"] == name]
        scores = [v for row in rows for v in row["per_game"]]
        summary[name] = {"mean": float(np.mean(scores)), "worst": float(np.min(scores))}
    selected = max(summary, key=lambda name: (summary[name]["worst"], summary[name]["mean"]))
    return {"records": records, "summary": summary, "selected_variant": selected}


def possession_rule_sweep(params, cfg, seeds=(0, 1, 2, 3), variants=None):
    """Paired experiment for team support pooling and latent-contest change confirmation."""
    variants = variants or {
        "off": {"control_team_support_weight": 0.0, "team_change_confirm_frames": 0},
        "top2_w20": {"control_team_support_weight": 0.2, "control_team_support_k": 2,
                     "team_change_confirm_frames": 0},
        "top3_w20": {"control_team_support_weight": 0.2, "control_team_support_k": 3,
                     "team_change_confirm_frames": 0},
        "confirm15": {"team_change_confirm_frames": 15,
                      "team_change_confirm_fraction": 0.65,
                      "team_change_min_strength": 0.15,
                      "team_change_min_margin": 0.05},
        "confirm20": {"team_change_confirm_frames": 20,
                      "team_change_confirm_fraction": 0.65,
                      "team_change_min_strength": 0.15,
                      "team_change_min_margin": 0.05},
        "top2_confirm15": {"control_team_support_weight": 0.2,
                           "control_team_support_k": 2,
                           "team_change_confirm_frames": 15,
                           "team_change_confirm_fraction": 0.65,
                           "team_change_min_strength": 0.15,
                           "team_change_min_margin": 0.05},
        "acquisition15": {"controller_acquisition_weight": 0.15,
                           "team_change_confirm_frames": 15,
                           "team_change_confirm_fraction": 0.65,
                           "team_change_min_strength": 0.15,
                           "team_change_min_margin": 0.05},
        "context10_acq15": {"controller_acquisition_weight": 0.15,
                             "team_change_confirm_frames": 10,
                             "team_change_confirm_fraction": 0.65,
                             "team_change_min_strength": 0.15,
                             "team_change_min_margin": 0.05,
                             "team_change_contextual": 1},
        "all_context15": {"control_team_support_weight": 0.2,
                          "control_team_support_k": 2,
                          "controller_acquisition_weight": 0.15,
                          "team_change_confirm_frames": 15,
                          "team_change_confirm_fraction": 0.65,
                          "team_change_min_strength": 0.15,
                          "team_change_min_margin": 0.05,
                          "team_change_contextual": 1},
    }
    records = []
    for seed in seeds:
        prepared = prepare(seed=seed)
        for name, update in variants.items():
            p = dict(params)
            p.update(update)
            mean, per_game = score(prepared, p, cfg)
            records.append({"seed": int(seed), "variant": name, "spell": mean,
                            "per_game": per_game})
    summary = {}
    for name in variants:
        rows = [r for r in records if r["variant"] == name]
        scores = [v for row in rows for v in row["per_game"]]
        summary[name] = {"mean": float(np.mean(scores)), "worst": float(np.min(scores))}
    selected = max(summary, key=lambda name: (summary[name]["worst"], summary[name]["mean"]))
    return {"records": records, "summary": summary, "selected_variant": selected}


def write_params(best, path, n_trials, seed, controller_stability=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    result = {"source": "random search on Metrica Sample Games 1-2", "seed": seed,
              "n_trials": n_trials, "spell_f1_metrica": best["per_game"],
              "params": best["params"], "decode": best["decode"]}
    if controller_stability is not None:
        result["controller_stability_validation"] = controller_stability
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    return path
