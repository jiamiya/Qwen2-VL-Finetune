import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from transformers import (
    AutoConfig, AutoProcessor,
    AutoModelForVision2Seq, AutoModelForCausalLM
)
from peft import PeftModel


def analyze_coordinate_string(s: str):
    s = s.strip()
    try:
        pairs_content = re.findall(r'\[([^\[\]]+)\]', s)
        full_pairs = re.findall(r'\[[^\[\]]+\]', s)
    except Exception:
        return False, "结构异常：内部方括号不匹配或格式混乱。"

    temp_s = s
    for pair_str in full_pairs:
        temp_s = temp_s.replace(pair_str, '', 1)
    remaining_chars = re.sub(r'[\s,]', '', temp_s)
    if remaining_chars == '[]':
        remaining_chars = ''
    if remaining_chars:
        return False, f"结构异常：在数对之外发现了多余的字符: '{remaining_chars}'"

    if len(pairs_content) != 10:
        return False, f"数量异常：发现了 {len(pairs_content)} 个数对，但需要正好 10 个。"

    for i, pair_str in enumerate(pairs_content):
        parts = pair_str.split(',')
        if len(parts) != 2:
            return False, f"数对 #{i+1} 格式异常：需要2个数字，但发现了 {len(parts)} 个部分。内容: '[{pair_str}]'"
        try:
            float(parts[0].strip()); float(parts[1].strip())
        except ValueError:
            return False, f"数值异常：数对 #{i+1} 中包含无法解析为数字的内容。内容: '[{pair_str}]'"
    return True, "格式正常"


def fix_coordinate_string(s: str):
    found_pairs = []
    for match in re.finditer(r'\[([^\[\]]+)\]', s):
        content = match.group(1)
        parts = [p.strip() for p in content.split(',')]
        if len(parts) == 2:
            try:
                num1 = float(parts[0]); num2 = float(parts[1])
                center_pos = (match.span()[0] + match.span()[1]) / 2
                found_pairs.append({'value': [num1, num2], 'span': match.span(), 'center': center_pos})
            except ValueError:
                continue

    if not found_pairs:
        default_pair = "[0.00, 0.00]"
        return f"[{', '.join([default_pair] * 10)}]", "修复操作：未找到任何有效数对，已生成默认字符串。"

    if len(found_pairs) >= 10:
        final_pairs_values = [p['value'] for p in found_pairs[:10]]
        fix_reason = f"修复操作：保留了前 10 个有效数对 (原先有 {len(found_pairs)} 个)。"
    else:
        final_pairs_values = [None] * 10
        string_length = len(s)

        if string_length < 10:
            if len(found_pairs) == 1:
                final_pairs_values[4] = found_pairs[0]['value']
            else:
                step = 9.0 / (len(found_pairs) - 1)
                for i, pair in enumerate(found_pairs):
                    pos = int(round(i * step))
                    while pos < 10 and final_pairs_values[pos] is not None:
                        pos += 1
                    if pos < 10:
                        final_pairs_values[pos] = pair['value']
        else:
            interval_length = string_length / 10.0
            position_mapping = {}
            for pair in found_pairs:
                logical_pos = int(round(pair['center'] / interval_length))
                logical_pos = min(max(logical_pos, 0), 9)
                position_mapping.setdefault(logical_pos, []).append(pair)

            conflicted_pairs = []
            for logical_pos, pairs_list in position_mapping.items():
                if len(pairs_list) == 1:
                    final_pairs_values[logical_pos] = pairs_list[0]['value']
                else:
                    final_pairs_values[logical_pos] = pairs_list[0]['value']
                    conflicted_pairs.extend(pairs_list[1:])

            for pair in conflicted_pairs:
                placed = False
                original_pos = int(round(pair['center'] / interval_length))
                original_pos = min(max(original_pos, 0), 9)
                for offset in range(10):
                    if original_pos + offset < 10 and final_pairs_values[original_pos + offset] is None:
                        final_pairs_values[original_pos + offset] = pair['value']; placed = True; break
                    if offset > 0 and original_pos - offset >= 0 and final_pairs_values[original_pos - offset] is None:
                        final_pairs_values[original_pos - offset] = pair['value']; placed = True; break

        num_missing = sum(1 for x in final_pairs_values if x is None)
        for i in range(10):
            if final_pairs_values[i] is None:
                prev_value = None; prev_idx = -1
                for j in range(i-1, -1, -1):
                    if final_pairs_values[j] is not None:
                        prev_value = final_pairs_values[j]; prev_idx = j; break
                next_value = None; next_idx = -1
                for j in range(i+1, 10):
                    if final_pairs_values[j] is not None:
                        next_value = final_pairs_values[j]; next_idx = j; break
                if prev_value and next_value:
                    ratio = (i - prev_idx) / (next_idx - prev_idx)
                    x = prev_value[0] + ratio * (next_value[0] - prev_value[0])
                    y = prev_value[1] + ratio * (next_value[1] - prev_value[1])
                    final_pairs_values[i] = [x, y]
                elif prev_value:
                    final_pairs_values[i] = prev_value[:]
                elif next_value:
                    final_pairs_values[i] = next_value[:]
                else:
                    final_pairs_values[i] = [0.0, 0.0]

        conflict_count = len(found_pairs) - sum(v is not None for v in final_pairs_values)
        if conflict_count > 0:
            fix_reason = f"修复操作：基于位置映射和插值的方式补齐了 {num_missing} 个数对（解决了 {conflict_count} 个位置冲突）。"
        else:
            fix_reason = f"修复操作：基于位置映射和插值的方式补齐了 {num_missing} 个数对。"

    reconstructed_pairs = [f"[{p[0]:.2f}, {p[1]:.2f}]" for p in final_pairs_values]
    final_string = f"[{', '.join(reconstructed_pairs)}]"
    return final_string, fix_reason


# ===============================
# 工具函数：解析文本为 (10,2) 数组
# ===============================

def to_pairs_10x2_from_text(text: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    ok, reason = analyze_coordinate_string(text)
    used_fix = False
    out_str = text
    if not ok:
        out_str, fix_reason = fix_coordinate_string(text)
        used_fix = True
        reason = fix_reason

    # 解析最终字符串为 (10,2)
    pairs_content = re.findall(r'\[([^\[\]]+)\]', out_str)
    vals = []
    for pair_str in pairs_content[:10]:
        a, b = [x.strip() for x in pair_str.split(',')]
        vals.append([float(a), float(b)])
    arr = np.array(vals, dtype=np.float32)
    if arr.shape != (10, 2):
        # 极端兜底
        pad = np.zeros((10, 2), dtype=np.float32)
        pad[:min(len(vals), 10)] = arr[:min(len(vals), 10)]
        arr = pad
    meta = {"ok": ok, "used_fix": used_fix, "reason": reason, "normalized_text": out_str}
    return arr, meta


# ===============================
# 加载 & 评估
# ===============================

def load_vlm_auto(
    model_id: str = None,
    base_model_id: str = None,
    adapter_dir: str = None,
    dtype: str = "bf16",
    device: str = "cuda",
    trust_remote_code: bool = True,
):
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

    def _load_base(mid):
        # 1) 先拿 config 决定该用哪个 Auto 类
        cfg = AutoConfig.from_pretrained(mid, trust_remote_code=trust_remote_code)
        # 经验判断：有 vision_config 或 model_type 属于常见 VLM，就用 Vision2Seq
        model_type = getattr(cfg, "model_type", "")
        looks_vlm = hasattr(cfg, "vision_config") or model_type in {
            "qwen2_5_vl", "qwen2vl", "llava", "fuyu", "phi3-vision", "mllama", "internvl", "minicpmv"
        }

        processor = AutoProcessor.from_pretrained(mid, trust_remote_code=trust_remote_code)

        # 2) 尝试优先用看起来合适的分支；失败再回退
        if looks_vlm:
            try:
                mdl = AutoModelForVision2Seq.from_pretrained(
                    mid, torch_dtype=torch_dtype, device_map=None, trust_remote_code=trust_remote_code
                )
            except Exception:
                mdl = AutoModelForCausalLM.from_pretrained(
                    mid, torch_dtype=torch_dtype, device_map=None, trust_remote_code=trust_remote_code
                )
        else:
            try:
                mdl = AutoModelForCausalLM.from_pretrained(
                    mid, torch_dtype=torch_dtype, device_map=None, trust_remote_code=trust_remote_code
                )
            except Exception:
                mdl = AutoModelForVision2Seq.from_pretrained(
                    mid, torch_dtype=torch_dtype, device_map=None, trust_remote_code=trust_remote_code
                )

        mdl.to(device).eval()
        return mdl, processor

    # ==== 显式 LoRA 适配器 ====
    if adapter_dir is not None:
        if base_model_id is None:
            raise ValueError("使用 LoRA 适配器时请提供 --base_model_id")
        if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            raise FileNotFoundError(f"{adapter_dir} 下未找到 adapter_config.json")
        base, processor = _load_base(base_model_id)
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).to(device).eval()
        return model, processor

    # ==== 单目录自动识别 ====
    if model_id is None:
        raise ValueError("至少提供 --model_id 或 (--base_model_id + --adapter_dir)")

    if os.path.isdir(model_id) and os.path.exists(os.path.join(model_id, "adapter_config.json")):
        if base_model_id is None:
            raise ValueError("检测到 LoRA 适配器目录，但未提供 --base_model_id")
        base, processor = _load_base(base_model_id)
        model = PeftModel.from_pretrained(base, model_id, is_trainable=False).to(device).eval()
        return model, processor

    # 普通完整模型（原生或已 merge）
    model, processor = _load_base(model_id)
    return model, processor



def build_messages_from_conversations(conv: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    把 {from:human/gpt, value: "... <image> ..."} 转成 apply_chat_template 用的 messages。
    - human -> role='user'
    - gpt   -> role='assistant'
    内容里如果包含 <image>，交给 processor 在 inputs 里处理；这里文本里保留占位就行。
    """
    role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
    msgs = []
    for turn in conv:
        role = role_map.get(turn.get("from", "human"), "user")
        content = turn["value"]
        # 简化做法：按原样放进去（多数 VLM 的 processor 会在 text + images 对齐时替换占位）
        msgs.append({"role": role, "content": [{"type": "text", "text": content}]})
    return msgs


@torch.no_grad()
def generate_pairs_for_item(
    model, processor, sample: Dict[str, Any], image_folder: str, device: str,
    max_new_tokens: int = 128, temperature: float = 0.0, top_p: float = 1.0
) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """
    对单条样本做生成 + 解析为 (10,2)。
    返回：parsed_pairs, meta, raw_text
    """
    # 载入图片
    img_path = sample["image"]
    if (not os.path.exists(img_path)) and (not img_path.startswith("http")):
        img_path = os.path.join(image_folder, img_path)
    image = Image.open(img_path).convert("RGB")

    # 构造 messages（用 conversations 原样）
    messages = build_messages_from_conversations(sample["conversations"])
    # 为多数 VLM 构建 chat 模板文本
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[chat_text], images=[image], return_tensors="pt").to(device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=(temperature > 0.0),
        use_cache=True,
    )
    out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    # 取 assistant 回答部分（有些模型会把 prompt 带回来），尽量截后半段
    # 简单策略：找到最后一个 "assistant" 标记或直接用全文
    raw = out_text.strip()

    pairs, meta = to_pairs_10x2_from_text(raw)
    return pairs, meta, raw


def ade_fde_xy(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, horizons_s: List[float], dt: float):
    """
    pred, gt: [N,2], mask: [N], 计算各 horizon 的 ADE/FDE
    """
    n_valid = int(mask.sum())
    if n_valid <= 0:
        return {h: {"ade": None, "fde": None} for h in horizons_s}

    out = {}
    for h in horizons_s:
        k = min(int(round(h / dt)), n_valid)
        if k <= 0:
            out[h] = {"ade": None, "fde": None}
            continue
        diff = pred[:k, :2] - gt[:k, :2]
        dists = np.linalg.norm(diff, axis=-1)
        ade = float(dists.mean())
        fde = float(np.linalg.norm(pred[k-1, :2] - gt[k-1, :2]))
        out[h] = {"ade": ade, "fde": fde}
    return out


def run_eval_vlm(
    model,
    processor,
    eval_json: str,
    image_folder: str,
    output_dir: str,
    device: str = "cuda",
    dt: float = 0.5,
    horizons: List[float] = [1.0, 2.0, 3.0],
    batch_show_progress: bool = True,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    # 读数据
    with open(eval_json, "r") as f:
        data = json.load(f)

    rows = []
    buckets = {h: {"ade": [], "fde": []} for h in horizons}

    for idx, sample in enumerate(data):
        try:
            pairs_pred, meta, raw_text = generate_pairs_for_item(
                model, processor, sample, image_folder=image_folder, device=device,
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
            )
            print(pairs_pred, '\n', raw_text)
        except Exception as e:
            print(f"[E] gen error at {idx}: {e}")
            pairs_pred = np.zeros((10, 2), dtype=np.float32)
            meta = {"ok": False, "used_fix": True, "reason": f"generation_error: {e}", "normalized_text": ""}

        # 拿 GT & mask
        gt = np.array(sample.get("action_targets", [[0.0, 0.0]] * 10), dtype=np.float32)
        msk = np.array(sample.get("action_mask", [1] * len(gt)), dtype=bool)
        # 截断到 10
        gt = gt[:10]; msk = msk[:10]
        if gt.shape != (10, 2):
            pad = np.zeros((10, 2), dtype=np.float32); pad[:min(len(gt),10)] = gt[:min(len(gt),10)]; gt = pad
        if msk.shape != (10,):
            padm = np.zeros((10,), dtype=bool); padm[:min(len(msk),10)] = msk[:min(len(msk),10)]; msk = padm

        metrics = ade_fde_xy(pairs_pred, gt, msk, horizons_s=horizons, dt=dt)

        # 记录逐样本
        for h in horizons:
            ade = metrics[h]["ade"]; fde = metrics[h]["fde"]
            if ade is not None: buckets[h]["ade"].append(ade)
            if fde is not None: buckets[h]["fde"].append(fde)
            rows.append({
                "index": idx,
                "id": sample.get("id", str(idx)),
                "scene": sample.get("scene", ""),
                "horizon_s": h,
                "ade": ade,
                "fde": fde,
                "parsed_ok": meta["ok"],
                "used_fix": meta["used_fix"],
                "reason": meta["reason"],
            })

        if batch_show_progress and ((idx + 1) % 20 == 0 or (idx + 1) == len(data)):
            print(f"[{idx+1}/{len(data)}] done")

    # 汇总
    summary = {}
    for h in horizons:
        ade_arr = np.array(buckets[h]["ade"], dtype=np.float64)
        fde_arr = np.array(buckets[h]["fde"], dtype=np.float64)
        summary[h] = {
            "ade_mean": float(ade_arr.mean()) if ade_arr.size else None,
            "ade_std": float(ade_arr.std(ddof=0)) if ade_arr.size else None,
            "fde_mean": float(fde_arr.mean()) if fde_arr.size else None,
            "fde_std": float(fde_arr.std(ddof=0)) if fde_arr.size else None,
            "count": int(ade_arr.size),
        }

    # 保存
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "vlm_metrics_per_sample.csv")
    df.to_csv(csv_path, index=False)

    js_path = os.path.join(output_dir, "vlm_metrics_summary.json")
    with open(js_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 直方图
    for h in horizons:
        ade_arr = np.array(buckets[h]["ade"], dtype=np.float64)
        fde_arr = np.array(buckets[h]["fde"], dtype=np.float64)
        if ade_arr.size:
            plt.figure(); plt.hist(ade_arr, bins=50); plt.title(f"VLM ADE @ {h:.1f}s"); plt.xlabel("ADE"); plt.ylabel("Count"); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"vlm_ade_hist_{h:.1f}s.png")); plt.close()
        if fde_arr.size:
            plt.figure(); plt.hist(fde_arr, bins=50); plt.title(f"VLM FDE @ {h:.1f}s"); plt.xlabel("FDE"); plt.ylabel("Count"); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"vlm_fde_hist_{h:.1f}s.png")); plt.close()

    print(f"[saved] {csv_path}\n[saved] {js_path}")
    for h, s in summary.items():
        print(f"{h:.1f}s: ADE {s['ade_mean']:.4f}±{s['ade_std']:.4f} | FDE {s['fde_mean']:.4f}±{s['fde_std']:.4f} | n={s['count']}")
    return summary, csv_path, js_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default=None, help="完整 VLM 模型（原生或已合并）")
    ap.add_argument("--base_model_id", type=str, default=None, help="LoRA 时的基座模型名")
    ap.add_argument("--adapter_dir", type=str, default=None, help="LoRA 适配器目录（含 adapter_config.json）")
    ap.add_argument("--trust_remote_code", action="store_true", help="需要时打开，兼容部分模型自定义代码")
    ap.add_argument("--eval_json", required=True, type=str, help="eval 数据 json 路径（与你训练时相同格式）")
    ap.add_argument("--image_folder", required=True, type=str, help="图像根目录（用于补相对路径）")
    ap.add_argument("--output_dir", required=True, type=str, help="评估结果输出目录")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--horizons", type=str, default="1,2,3")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    args = ap.parse_args()
    
    
    # 加载模型与处理器
    model, processor = load_vlm_auto(
        model_id=args.model_id,
        base_model_id=args.base_model_id,
        adapter_dir=args.adapter_dir,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

    horizons = [float(x) for x in args.horizons.split(",")]
    run_eval_vlm(
        model,
        processor,
        eval_json=args.eval_json,
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        device=args.device,
        dt=args.dt,
        horizons=horizons,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
