import os
import json
import re
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from transformers import AutoProcessor

# === 1) 你的项目内的模块（按你的实际路径修改 import） ===
from src.dataset.vla_dataset import SupervisedVLADataset as SupervisedDataset
from src.dataset.vla_dataset import DataCollatorForSupervisedVLADataset as DataCollatorForSupervisedDataset  # 这是你之前贴的 dataset/collator 文件路径
from src.model.qwen2_5_vla import QwenVLAWrapper  # 你的 wrapper 类路径
from peft import PeftModel


def _tensor_fingerprint(t: torch.Tensor):
    """给张量做一个轻量指纹，便于对比加载前后是否变化。"""
    try:
        with torch.no_grad():
            # 统一到 CPU float32 做稳定的指纹
            x = t.detach().to("cpu", dtype=torch.float32).flatten()
            # 选取均值、方差、L1/L2 作为多元指纹，避免仅用 sum 碰撞
            mean = x.mean().item()
            std = x.std(unbiased=False).item()
            l1 = x.abs().mean().item()
            l2 = (x.pow(2).mean().sqrt()).item()
            n = x.numel()
        return {"n": n, "mean": mean, "std": std, "l1": l1, "l2": l2}
    except Exception as e:
        return {"err": str(e)}

def _collect_action_params(named_params, patterns):
    """按给定正则/字符串匹配参数名，返回 {name: tensor}。"""
    selected = {}
    for name, p in named_params:
        for pat in patterns:
            if isinstance(pat, str):
                ok = pat in name
            else:
                ok = bool(pat.search(name))
            if ok:
                selected[name] = p
                break
    return selected

def _print_action_params(title, params_dict):
    print(f"\n[{title}] 共 {len(params_dict)} 个 action 相关参数：")
    if not params_dict:
        print("（未找到）")
        return
    for k, v in params_dict.items():
        try:
            print(f"  - {k:50s} shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}, requires_grad={v.requires_grad}")
        except Exception as e:
            print(f"  - {k:50s} <打印失败: {e}>")

def _print_fingerprints(title, fp_before, fp_after):
    print(f"\n[{title}] 指纹对比（加载 non-LoRA 前 -> 后）：")
    if not fp_before:
        print("（无“前”指纹，可忽略）")
        return
    keys = sorted(set(fp_before.keys()) | set(fp_after.keys()))
    for k in keys:
        b = fp_before.get(k)
        a = fp_after.get(k)
        if b is None:
            print(f"  + {k}: 仅出现在“后”（可能是新加参数/已从别处映射）")
        elif a is None:
            print(f"  - {k}: 仅出现在“前”（不太常见，检查加载逻辑）")
        else:
            changed = ("err" in b or "err" in a) or any(abs(b[m]-a[m]) > 1e-12 for m in ("mean","std","l1","l2")) or (b["n"] != a["n"])
            tag = "CHANGED" if changed else "unchanged"
            print(f"  * {k}: {tag} | n={b.get('n')} -> {a.get('n')}, mean={b.get('mean'):.6f} -> {a.get('mean'):.6f}, std={b.get('std'):.6f} -> {a.get('std'):.6f}")

def load_model_and_processor(
    base_model_id: str,
    ckpt_dir: str,
    dtype: str = "bf16",
    device: str = "cuda",
    expected_action_keys=[],
):
    """加载 Wrapper 基座 + LoRA 适配器 + non_lora_state_dict（含 action_*），并打印 action 参数加载校验信息。"""
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

    # ---- 0) 匹配规则（默认：名字里包含 'action' 或以 'action_' 开头） ----
    if expected_action_keys is None:
        action_matchers = ["action", re.compile(r"(^|[.])action(_|[.])")]
    else:
        # 既用用户指定关键字，也保留通用 action 模糊匹配，防漏报
        extra = ["action", re.compile(r"(^|[.])action(_|[.])")]
        action_matchers = list(expected_action_keys) + extra

    # 1) 基座（wrapper）
    base = QwenVLAWrapper.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(device)

    # 2) LoRA 适配器
    model = PeftModel.from_pretrained(base, ckpt_dir, is_trainable=False).to(device)
    model.eval()

    # 2.1) 抓取“加载 non-LoRA 之前”的 action 参数指纹
    target = model.get_base_model() if hasattr(model, "get_base_model") else model.base_model.model
    before_params = _collect_action_params(target.named_parameters(), action_matchers)
    _print_action_params("加载 non-LoRA 之前的 action 参数", before_params)
    fp_before = {k: _tensor_fingerprint(v) for k, v in before_params.items()}

    # 3) 非 LoRA 权重（包括 action_token_embeds / action_head）
    nl_path = os.path.join(ckpt_dir, "non_lora_state_dict.bin")
    if os.path.exists(nl_path):
        state = torch.load(nl_path, map_location="cpu")
        print(f"\n[load non-LoRA] 原始条目数: {len(state)}")

        # --- 关键：规范化键名前缀 ---
        def strip_prefix(k: str):
            # 常见包裹前缀按需裁剪；可按实际再补充
            prefixes = [
                "base_model.model.",
                "model.",
                "module.",
                "wrapped.",
            ]
            for p in prefixes:
                if k.startswith(p):
                    return k[len(p):]
            return k

        new_state = {}
        changed = 0
        for k, v in state.items():
            nk = strip_prefix(k)
            new_state[nk] = v
            if nk != k:
                changed += 1
        print(f"[load non-LoRA] 规范化键名：{changed} / {len(state)} 个键被去前缀")

        # 打印 action 相关键（规范化后）
        action_state = {k: v for k, v in new_state.items() if "action" in k}
        print(f"[load non-LoRA] 规范化后与 action 相关的条目：{len(action_state)}")
        for k, v in list(action_state.items())[:20]:
            shape = tuple(v.shape) if hasattr(v, "shape") else "NA"
            print(f"  - {k:50s} shape={shape}")
        if len(action_state) > 20:
            print(f"  ...（其余 {len(action_state)-20} 条省略）")

        # 真正加载
        missing, unexpected = target.load_state_dict(new_state, strict=False)
        print(f"\n[load non-LoRA] missing_total={len(missing)} unexpected_total={len(unexpected)}")

        # 聚焦 action 相关
        missing_action = [k for k in missing if "action" in k]
        unexpected_action = [k for k in unexpected if "action" in k]
        if missing_action:
            print(f"[load non-LoRA] ⚠ 与 action 相关的缺失键（missing）{len(missing_action)}：")
            for k in missing_action: print(f"  - {k}")
        if unexpected_action:
            print(f"[load non-LoRA] ⚠ 与 action 相关的多余键（unexpected）{len(unexpected_action)}：")
            for k in unexpected_action: print(f"  - {k}")
        if not missing_action and not unexpected_action:
            print("[load non-LoRA] ✓ action 相关键已正确对齐。")

    # 4) processor
    processor = AutoProcessor.from_pretrained(base_model_id)
    return model, processor


def make_eval_loader(
    eval_json: str,
    processor,
    image_folder: str,
    batch_size: int = 8,
    num_workers: int = 2,
):
    """复用训练时的数据管道，返回 DataLoader。"""
    # 复用 DataArguments 中用到的关键字段，通过简单对象伪装一下
    class _DA:
        def __init__(self, data_path, image_folder):
            self.data_path = data_path
            self.image_folder = image_folder
            # 下面这些值用你的训练默认；可按需改
            self.image_min_pixels = 256 * 28 * 28
            self.image_max_pixels = 1280 * 28 * 28
            self.video_min_pixels = 0
            self.video_max_pixels = 0
            self.image_resized_width = None
            self.image_resized_height = None
            self.video_resized_width = None
            self.video_resized_height = None
            self.fps = 2.0  # 这里仅占位（不用于静态图）
    data_args = _DA(eval_json, image_folder)

    # 这里 model_id 仅用于数据管道中的条件分支（你那边有 "Qwen2.5" 判定）；直接给 base 模型名
    eval_dataset = SupervisedDataset(
        data_path=eval_json,
        processor=processor,
        data_args=data_args,
        model_id="Qwen2.5",
        padding=True,
    )
    collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    return loader


@torch.no_grad()
def run_eval(
    model,
    loader,
    device: str,
    horizons_s: List[float] = [1.0, 2.0, 3.0],
    dt: float = 0.5,   # 你的数据间隔 0.5s
    save_dir: str = "./eval_out",
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    # 对每个 horizon 统计所有样本的 ADE/FDE
    bucket = {h: {"ade": [], "fde": []} for h in horizons_s}

    for step, batch in tqdm(enumerate(loader), total=len(loader)):
        # if step>20:
        #     break
        # 移动到 device
        batch_on_device = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch_on_device[k] = v.to(device)
            else:
                batch_on_device[k] = v

        # 关闭 LM loss，只要动作预测
        batch_on_device["compute_lm_loss"] = False
        batch_on_device["output_hidden_states"] = True  # 确保 wrapper 能取到最后层 hidden

        # start_t = time.time()
        out = model(**batch_on_device)
        # print('inference time: ', time.time()-start_t)
        # 形状：[B, N, A]，A=2（x,y）或（speed,curvature）
        pred = out.action_preds
        if pred is None:
            raise RuntimeError("Model output has no action_preds. Check wrapper forward.")

        # 真值/掩码
        gt = batch_on_device.get("action_targets", None)
        msk = batch_on_device.get("action_mask", None)
        if (gt is None) or (msk is None):
            raise RuntimeError("Eval loader must provide action_targets and action_mask.")

        # 对齐到同 dtype/cpu 做 numpy 计算
        pred_np = pred.detach().float().cpu().numpy()
        gt_np = gt.detach().float().cpu().numpy()
        msk_np = msk.detach().bool().cpu().numpy()
        print("pred: ", pred_np[0])
        print("gt: ", gt_np[0])
        # print('loss: ', out.action_loss)

        B, N, A = pred_np.shape
        # 逐样本计算
        for b in range(B):
            n_valid = int(msk_np[b].sum())
            if n_valid == 0:
                continue
            # 截到有效长度
            p = pred_np[b, :n_valid, :]  # [n, A]
            g = gt_np[b, :n_valid, :]    # [n, A]

            # ADE/FDE 按每个 horizon 计算（A=2 时为欧氏距离；若 A>2，可只取前2维或改 metric）
            for h in horizons_s:
                k = min(int(round(h / dt)), n_valid)  # horizon 对应的步数（最多到 n_valid）
                if k <= 0:
                    continue

                # 采用前两维做 xy 误差；若是 speed/curvature，你也可以把这里换成 L2 over 全维
                diff = p[:k, :2] - g[:k, :2]            # [k, 2]
                dists = np.linalg.norm(diff, axis=-1)   # [k]
                ade = float(dists.mean())
                fde = float(np.linalg.norm((p[k-1, :2] - g[k-1, :2])))
                print(f"{h} s metrics: {ade, fde}")

                bucket[h]["ade"].append(ade)
                bucket[h]["fde"].append(fde)

                all_rows.append({
                    "index": step * B + b,
                    "n_valid": n_valid,
                    "horizon_s": h,
                    "ade": ade,
                    "fde": fde,
                })

    # 汇总统计
    summary = {}
    for h in horizons_s:
        ade_arr = np.array(bucket[h]["ade"], dtype=np.float64)
        fde_arr = np.array(bucket[h]["fde"], dtype=np.float64)
        summary[h] = {
            "ade_mean": float(ade_arr.mean()) if ade_arr.size > 0 else None,
            "ade_std": float(ade_arr.std(ddof=0)) if ade_arr.size > 0 else None,
            "fde_mean": float(fde_arr.mean()) if fde_arr.size > 0 else None,
            "fde_std": float(fde_arr.std(ddof=0)) if fde_arr.size > 0 else None,
            "count": int(ade_arr.size),
        }

    # 保存明细 CSV
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(save_dir, "metrics_per_sample.csv")
    df.to_csv(csv_path, index=False)

    # 保存 summary JSON
    js_path = os.path.join(save_dir, "metrics_summary.json")
    with open(js_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {csv_path}\n[saved] {js_path}")

    # 画直方图
    for h in horizons_s:
        ade_arr = np.array(bucket[h]["ade"], dtype=np.float64)
        fde_arr = np.array(bucket[h]["fde"], dtype=np.float64)
        if ade_arr.size == 0:
            continue

        plt.figure()
        plt.hist(ade_arr, bins=50)
        plt.title(f"ADE histogram @ {h:.1f}s")
        plt.xlabel("ADE")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"ade_hist_{h:.1f}s.png"))
        plt.close()

        plt.figure()
        plt.hist(fde_arr, bins=50)
        plt.title(f"FDE histogram @ {h:.1f}s")
        plt.xlabel("FDE")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"fde_hist_{h:.1f}s.png"))
        plt.close()

    return summary, csv_path, js_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", type=str, required=True,
                    help="原始基座模型（训练时用的底座），如 Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--ckpt_dir", type=str, required=True,
                    help="训练输出目录或 checkpoint-XXXX 目录（须包含 LoRA 适配器与 non_lora_state_dict.bin）")
    ap.add_argument("--eval_json", type=str, required=True,
                    help="eval 数据 json 路径（与你训练时相同格式）")
    ap.add_argument("--image_folder", type=str, required=True,
                    help="图像根目录（dataset 会用它补全相对路径）")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="评估结果输出目录")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dt", type=float, default=0.5, help="相邻点的时间间隔（秒）")
    ap.add_argument("--horizons", type=str, default="1,2,3", help="以逗号分隔的秒数列表，例如 1,2,3")
    args = ap.parse_args()

    horizons_s = [float(x) for x in args.horizons.split(",")]

    model, processor = load_model_and_processor(
        base_model_id=args.base_model_id,
        ckpt_dir=args.ckpt_dir,
        dtype=args.dtype,
        device=args.device,
    )

    loader = make_eval_loader(
        eval_json=args.eval_json,
        processor=processor,
        image_folder=args.image_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    summary, csv_path, js_path = run_eval(
        model=model,
        loader=loader,
        device=args.device,
        horizons_s=horizons_s,
        dt=args.dt,
        save_dir=args.output_dir,
    )

    print("\n==== Summary ====")
    for h, s in summary.items():
        print(f"{h:.1f}s: ADE mean={s['ade_mean']:.4f}±{s['ade_std']:.4f} | FDE mean={s['fde_mean']:.4f}±{s['fde_std']:.4f} | count={s['count']}")


if __name__ == "__main__":
    main()
