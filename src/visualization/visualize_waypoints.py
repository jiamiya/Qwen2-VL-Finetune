#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot trajectories + endpoint histograms, and compute per-step stats.

新增: 统计时间差分 dx, dy 的分布 (第一个点与原点差分).
"""

import argparse
import json
import os
from typing import List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import csv


def load_trajs_and_endpoints(json_path: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """加载所有未来轨迹（每条 [Ki,2]）与终点 [N,2]。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    trajs, endpoints = [], []
    for item in data:
        wp = item.get("future_waypoints") or item.get("action_targets")
        if wp is None:
            continue
        wpa = np.asarray(wp, dtype=float)
        if wpa.ndim != 2 or wpa.shape[1] != 2:
            continue

        mask = item.get("action_mask")
        if mask is not None and len(mask) == len(wpa):
            valid_idx = np.where(np.asarray(mask) == 1)[0]
            if len(valid_idx) > 0:
                wpa = wpa[:valid_idx[-1] + 1]
            else:
                continue

        trajs.append(wpa)
        endpoints.append((wpa[-1, 0], wpa[-1, 1]))

    endpoints_arr = np.asarray(endpoints, dtype=float) if endpoints else np.zeros((0, 2), dtype=float)
    return trajs, endpoints_arr


def plot_trajectories(trajs: List[np.ndarray], save_path: str, max_trajs=0, alpha=0.08, dpi=150):
    plt.figure()
    count = 0
    for t in trajs:
        if max_trajs and count >= max_trajs:
            break
        if t.shape[0] == 0:
            continue
        plt.plot(t[:, 0], t[:, 1], linewidth=0.6, alpha=alpha)
        count += 1
    plt.xlabel("x (m, forward)")
    plt.ylabel("y (m, left)")
    plt.title(f"Trajectory distribution (N={count})")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_hist(data, save_path, bins=100, xlabel="", title="", dpi=150):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def compute_per_step_stats(trajs: List[np.ndarray]) -> Dict[str, Any]:
    """统计每个时间步的 x,y 及差分 dx,dy 的统计量."""
    if not trajs:
        return {"per_step": [], "meta": {"num_trajs": 0, "max_horizon": 0}}

    K = max(t.shape[0] for t in trajs)
    N = len(trajs)

    # 填充 [N, K, 2]
    arr = np.full((N, K, 2), np.nan)
    for i, t in enumerate(trajs):
        arr[i, :t.shape[0], :] = t

    per_step = []
    for k in range(K):
        xk = arr[:, k, 0]
        yk = arr[:, k, 1]

        # dx, dy：第一个点和原点差分
        if k == 0:
            dxk = xk.copy()
            dyk = yk.copy()
        else:
            dxk = arr[:, k, 0] - arr[:, k - 1, 0]
            dyk = arr[:, k, 1] - arr[:, k - 1, 1]

        def stat(v):
            return {
                "min": float(np.nanmin(v)),
                "p1": float(np.nanpercentile(v, 1)),
                "mean": float(np.nanmean(v)),
                "var": float(np.nanvar(v)),
                "p99": float(np.nanpercentile(v, 99)),
                "max": float(np.nanmax(v)),
                "count": int(np.sum(~np.isnan(v))),
            }

        stats_k = {
            "step": k + 1,
            "x": stat(xk),
            "y": stat(yk),
            "dx": stat(dxk),
            "dy": stat(dyk),
        }
        per_step.append(stats_k)

    return {"per_step": per_step, "meta": {"num_trajs": N, "max_horizon": K}}


def save_stats_csv(stats: Dict[str, Any], csv_path: str):
    fields = ["step"]
    for key in ["x", "y", "dx", "dy"]:
        fields += [f"{key}_{m}" for m in ["min","p1","mean","var","p99","max","count"]]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in stats["per_step"]:
            line = {"step": row["step"]}
            for key in ["x","y","dx","dy"]:
                for m in ["min","p1","mean","var","p99","max","count"]:
                    line[f"{key}_{m}"] = row[key][m]
            writer.writerow(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--outdir", "-o", default="")
    ap.add_argument("--max-trajs", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.08)
    args = ap.parse_args()

    outdir = args.outdir.strip() or os.path.join(os.path.dirname(os.path.abspath(args.input)), "figs")
    os.makedirs(outdir, exist_ok=True)

    trajs, endpoints = load_trajs_and_endpoints(args.input)
    if not trajs:
        raise RuntimeError("No trajectories found.")

    # 轨迹分布
    plot_trajectories(trajs, os.path.join(outdir, "trajectory_distribution.png"),
                      max_trajs=args.max_trajs, alpha=args.alpha, dpi=args.dpi)

    # 终点分布
    plot_hist(endpoints[:,0], os.path.join(outdir, "endpoint_hist_x.png"),
              bins=args.bins, xlabel="endpoint x (m, forward)",
              title="Endpoint x distribution", dpi=args.dpi)
    plot_hist(endpoints[:,1], os.path.join(outdir, "endpoint_hist_y.png"),
              bins=args.bins, xlabel="endpoint y (m, left)",
              title="Endpoint y distribution", dpi=args.dpi)

    # 统计
    stats = compute_per_step_stats(trajs)
    with open(os.path.join(outdir,"stats_per_step.json"),"w",encoding="utf-8") as f:
        json.dump(stats,f,ensure_ascii=False,indent=2)
    save_stats_csv(stats, os.path.join(outdir,"stats_per_step.csv"))

    print("[ok] Figures and stats saved to", outdir)


if __name__ == "__main__":
    main()
