#!/usr/bin/env python3
"""
TensorBoard events文件可视化脚本
用法: python visualize_events.py [log_dir] [output_dir]
"""

import sys
import argparse
from pathlib import Path

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def parse_tensorboard_logs(log_dir):
    """解析TensorBoard日志文件"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # 获取所有标量标签
    tags = event_acc.Tags()['scalars']
    print(f"找到的标量标签: {tags}")
    
    data = {}
    for tag in tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data

def plot_tensorboard_data(data, save_dir="./plots"):
    """绘制并保存图表"""
    os.makedirs(save_dir, exist_ok=True)
    
    for tag, values in data.items():
        plt.figure(figsize=(10, 6))
        plt.plot(values['steps'], values['values'])
        plt.title(tag)
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.grid(True)
        
        # 保存图片
        filename = f"{tag.replace('/', '_')}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='可视化TensorBoard events文件')
    parser.add_argument('--log-dir', default='./logs', help='包含events文件的目录')
    parser.add_argument('--output-dir', default=None, help='输出图片的目录')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.log_dir
    
    # 确保tensorboard依赖可用
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("请先安装tensorboard: pip install tensorboard")
        return
    
    data = parse_tensorboard_logs(args.log_dir)
    plot_tensorboard_data(data, args.output_dir)
    
    print(f"✅ 可视化完成！图表保存在: {args.output_dir}")

if __name__ == "__main__":
    main()