import os
import numpy as np
from math import atan2
import json

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

# --------- 配置 ----------
VERSION = 'v1.0-trainval'
DATAROOT = '/scratch/gilbreth/cancui/data/nuscenes/full'
# 输出路径（分别为 train / val）
PROMPT_TYPE = ["long", "short"][0]
OUTPUT_JSON_TRAIN = f"/scratch/gilbreth/jmingyan/project/Qwen2-VL-Finetune/data/nuscenes_waypoint_{PROMPT_TYPE}_prompt_train.json"
OUTPUT_JSON_VAL   = f"/scratch/gilbreth/jmingyan/project/Qwen2-VL-Finetune/data/nuscenes_waypoint_{PROMPT_TYPE}_prompt_val.json"

# --------- 初始化 ----------
nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
scenes = nusc.scene

# 官方 train/val 场景名称列表
splits = create_splits_scenes()
train_scene_names = set(splits['train'])
val_scene_names   = set(splits['val'])

def convert_with_format(num):
    return "{:012d}".format(num)

# ===== 四元数 -> yaw（z 轴朝向） =====
def yaw_from_quat(q):
    """
    q: [w, x, y, z] (nuScenes 存储顺序)
    返回平面航向角 yaw（弧度），右手系，x 前 y 左 z 上
    """
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return atan2(siny_cosp, cosy_cosp)

# ===== 世界坐标 -> t0 自车系 =====
def world_to_ego0_xy(p_world_xy, p0_world_xy, yaw0):
    """
    将全局平面点 p_world_xy 投到 t0 自车系（原点 p0_world_xy，朝向 yaw0）。
    x 前进，y 向左。
    """
    dx = p_world_xy[0] - p0_world_xy[0]
    dy = p_world_xy[1] - p0_world_xy[1]
    cy, sy = np.cos(-yaw0), np.sin(-yaw0)  # 乘以 R(-yaw0) = R(yaw0)^T
    x_ego = cy * dx - sy * dy
    y_ego = sy * dx + cy * dy
    return float(x_ego), float(y_ego)

# ===== 文本化（如需继续保留对话字段）=====
def format_waypoints_str(pts_xy, prec=2):
    def fmt(v): 
        return f"{v:.{prec}f}"
    return ", ".join([f"[{fmt(x)}, {fmt(y)}]" for x, y in pts_xy])

def history_waypoints_ego0(obs_traj_world, yaw0, p0_xy, prec=2):
    hist = []
    for p in obs_traj_world:
        x, y = world_to_ego0_xy((p[0], p[1]), p0_xy, yaw0)
        hist.append([x, y])
    return hist

# --------- 结果容器（train / val） ----------
data_train = []
data_val = []
idx_train = 1
idx_val = 1

# --------- 主循环 ----------
for scene in scenes:
    name = scene['name']
    # 仅处理属于 train 或 val 的 scene；其他（如 test）跳过
    target_split = None
    if name in train_scene_names:
        target_split = 'train'
    elif name in val_scene_names:
        target_split = 'val'
    else:
        continue

    token = scene['token']
    first_sample_token = scene['first_sample_token']
    last_sample_token  = scene['last_sample_token']
    description = scene['description']

    # 收集该 scene 中所有关键帧的：前相机图像路径、ego pose、相机参数
    front_camera_images = []
    ego_poses = []
    camera_params = []
    curr_sample_token = first_sample_token

    while True:
        sample = nusc.get('sample', curr_sample_token)
        cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])

        # 你的路径替换规则保留
        front_camera_images.append(
            os.path.join(nusc.dataroot, cam_front_data['filename']).replace('/media','/scratch/gilbreth/cancui')
        )

        pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
        ego_poses.append(pose)

        camera_params.append(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']))

        if curr_sample_token == last_sample_token:
            break
        curr_sample_token = sample['next']

    scene_length = len(front_camera_images)
    print(f"[{target_split}] Scene {name} has {scene_length} frames")

    if scene_length < TTL_LEN:
        print(f"[{target_split}] Scene {name} has less than {TTL_LEN} frames, skipping...")
        continue

    # 全局轨迹（仅用到 x,y）
    ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]

    for i in range(scene_length - TTL_LEN + 1):
        # 观测段与未来段（关键帧间隔约 0.5s）
        obs_images = front_camera_images[i:i+OBS_LEN]
        obs_ego_traj_world = ego_traj_world[i:i+OBS_LEN]
        fut_ego_traj_world = ego_traj_world[i+OBS_LEN:i+TTL_LEN]

        # t0 = 观测末帧
        t0_pose = ego_poses[i + OBS_LEN - 1]
        p0 = t0_pose['translation']  # [x, y, z]
        yaw0 = yaw_from_quat(t0_pose['rotation'])
        p0_xy = (float(p0[0]), float(p0[1]))

        # 历史（如需写入 prompt，可保留；否则可删除以下两行）
        hist_waypoints = history_waypoints_ego0(obs_ego_traj_world, yaw0, p0_xy, prec=2)
        hist_waypoints_str = format_waypoints_str(hist_waypoints)

        # 未来 waypoint（监督信号）：将未来帧投到 t0 自车系
        fut_waypoints_ego = []
        for p in fut_ego_traj_world:
            xk, yk = world_to_ego0_xy((p[0], p[1]), p0_xy, yaw0)
            fut_waypoints_ego.append([xk, yk])
        fut_waypoints_str = format_waypoints_str(fut_waypoints_ego)

        if PROMPT_TYPE == "short":
            system_message = (
                f"Predict the next {FUT_LEN} ego-frame waypoints (x-forward, y-left) at 0.5s intervals based on front-view image."
            )
            user_prompt = ""
        elif PROMPT_TYPE == "long":
            system_message = (
                "You are an autonomous driving labeller. You are given a sequence of front-view images, "
                "the 5-second historical ego waypoints in the *current ego frame* (x-forward, y-left), "
                "and a driving rationale. Your task is to predict the future ego-frame waypoints for the next "
                f"{FUT_LEN} timesteps (≈0.5 s interval). Provide raw coordinates only.\n"
            )
            user_prompt = (
                "This is a frame from a front camera video captured.\n\n"
                f"The 5-second historical ego-frame waypoints (relative to the last observed frame) are {hist_waypoints_str}.\n\n"
                f"Generate the predicted future waypoints in the format [x_1, y_1], [x_2, y_2], ..., [x_{FUT_LEN}, y_{FUT_LEN}]. "
                "Write the raw text, not markdown or LaTeX. Future waypoints:\n    "
            )
            

        # 样本项（注意：这里只保留末帧作为输入图片）
        cur_data = {
            'id': convert_with_format(idx_train if target_split == 'train' else idx_val),
            'image': obs_images[-1],
            "action_targets": fut_waypoints_ego,
            "action_mask": [1] * len(fut_waypoints_ego),
            "scene": name,
            "conversations": [
                {
                    "from": "human",
                    "value": system_message + '<image>' + user_prompt
                },
                {
                    "from": "gpt",
                    "value": fut_waypoints_str
                }
            ]
        }

        if target_split == 'train':
            data_train.append(cur_data)
            idx_train += 1
        else:
            data_val.append(cur_data)
            idx_val += 1

# --------- 写出 ----------
os.makedirs(os.path.dirname(OUTPUT_JSON_TRAIN), exist_ok=True)
with open(OUTPUT_JSON_TRAIN, 'w', encoding='utf-8') as f:
    json.dump(data_train, f, indent=2, ensure_ascii=False)
with open(OUTPUT_JSON_VAL, 'w', encoding='utf-8') as f:
    json.dump(data_val, f, indent=2, ensure_ascii=False)

print(f"[ok] wrote train samples: {len(data_train)} -> {OUTPUT_JSON_TRAIN}")
print(f"[ok] wrote val   samples: {len(data_val)}   -> {OUTPUT_JSON_VAL}")
