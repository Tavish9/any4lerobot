"""
2025.4.15
To solve the data set frequency requirements of different strategies and avoid repeated data collection
kwj
"""
import logging
import time
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import cv2
import numpy as np
import pandas as pd
from scipy import signal
from pprint import pformat
from tqdm import tqdm

def check_and_create_dirs(src_root, dst_root):
    """检查目录并创建目标目录结构"""
    if not Path(src_root).exists():
        raise FileNotFoundError(f"源目录 {src_root} 不存在")
    
    dst_path = Path(dst_root)
    dst_path.mkdir(parents=True, exist_ok=True)
    return dst_path

def update_metadata(src_meta_path, dst_meta_path):
    """更新元数据文件"""
    # 加载原 info.json
    with open(src_meta_path / "info.json", "r") as f:
        info = json.load(f)
    
    # 修改为 10Hz
    info["fps"] = 10
    
    # 更新episodes信息（每个episode的帧数需要调整）
    with open(src_meta_path / "episodes.jsonl", "r") as f:
        episodes = [json.loads(line) for line in f]
    
    for ep in episodes:
        ep["length"] = ep["length"] // 3  # 降采样到1/3
    
    # 保存到新数据集
    dst_meta_path.mkdir(parents=True, exist_ok=True)
    with open(dst_meta_path / "info.json", "w") as f:
        json.dump(info, f)
    
    with open(dst_meta_path / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

# def compute_stats(dst_data_dir):
#     """计算统计信息"""
#     all_actions = []
#     all_states = []
    
#     for parquet_path in Path(dst_data_dir).glob("**/*.parquet"):
#         df = pd.read_parquet(parquet_path)
#         all_actions.extend(df["action"].values.tolist())
#         all_states.extend(df["state"].values.tolist())
    
#     stats = {
#         "action": {
#             "mean": np.mean(all_actions, axis=0).tolist(),
#             "std": np.std(all_actions, axis=0).tolist()
#         },
#         "state": {
#             "mean": np.mean(all_states, axis=0).tolist(),
#             "std": np.std(all_states, axis=0).tolist()
#         }
#     }
    
#     with open(Path(dst_data_dir).parent / "meta" / "stats.json", "w") as f:
#         json.dump(stats, f, indent=2)
def compute_stats(dst_data_dir):
    """计算统计信息"""
    stats = {}
    
    for parquet_path in Path(dst_data_dir).glob("**/*.parquet"):
        df = pd.read_parquet(parquet_path)
        for col in ['action', 'observation.state']:  # 只处理指定特征列
            if col not in df.columns:
                continue
                
            if col not in stats:
                stats[col] = {'values': []}
            
            stats[col]['values'].extend(df[col].values.tolist())
    
    # 计算统计量
    final_stats = {}
    for col, data in stats.items():
        final_stats[col] = {
            "mean": np.mean(data['values'], axis=0).tolist(),
            "std": np.std(data['values'], axis=0).tolist()
        }
    
    with open(Path(dst_data_dir).parent / "meta" / "stats.json", "w") as f:
        json.dump(final_stats, f, indent=2)

# def downsample_parquet(episode_path, output_path):
#     """降采样Parquet文件"""
#     df = pd.read_parquet(episode_path)
    
#     # # 使用抗混叠滤波 + 降采样（30Hz->10Hz）
#     # downsampled_action = signal.decimate(df["action"].values, 3, axis=0, ftype='fir')
#     # downsampled_state = signal.decimate(df["state"].values, 3, axis=0, ftype='fir')
    
#     # 直接降采样
#     downsampled_action = df["action"].values[::3]
#     downsampled_state = df["observation.state"].values[::3]
    
#     downsampled_df = pd.DataFrame({
#         "action": list(downsampled_action),
#         "observation.state": list(downsampled_state),
#         "timestamp": df["timestamp"].values[::3]  # 时间戳同步处理
#     })
    
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     downsampled_df.to_parquet(output_path)
def downsample_parquet(episode_path, output_path):
    """降采样Parquet文件"""
    df = pd.read_parquet(episode_path)
    
    # 定义需要处理的特征列
    feature_columns = ['action', 'observation.state']
    
    # 定义需要排除的列
    exclude_columns = ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
    
    downsampled_data = {}
    for col in feature_columns:
        if col not in df.columns:
            raise KeyError(f"关键列 {col} 不存在于数据中，请检查数据结构")
        
        # 直接降采样（30Hz->10Hz）
        downsampled = df[col].values[::3]
        downsampled_data[col] = list(downsampled)
    
    # 保留必要元数据
    for meta_col in ['timestamp', 'frame_index', 'episode_index']:
        if meta_col in df.columns:
            downsampled_data[meta_col] = df[meta_col].values[::3]
    
    # 确保时间戳是连续的
    if 'timestamp' in downsampled_data:
        # 重新生成时间戳（可选）
        downsampled_data['timestamp'] = np.arange(len(downsampled_data['timestamp'])) * (1 / 10)  # 10Hz
        # 或者校正时间戳
        prev_ts = None
        corrected_timestamps = []
        for ts in downsampled_data['timestamp']:
            if prev_ts is not None and ts <= prev_ts:
                ts = prev_ts + 0.1  # 假设10Hz，时间间隔为0.1秒
            corrected_timestamps.append(ts)
            prev_ts = ts
        downsampled_data['timestamp'] = corrected_timestamps
    
    downsampled_df = pd.DataFrame(downsampled_data)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    downsampled_df.to_parquet(output_path)

def downsample_video(input_path, output_path, target_fps=10):
    """降采样视频文件，支持动态帧率调整和错误处理
    
    Args:
        input_path (Path): 输入视频文件路径
        output_path (Path): 输出视频文件路径
        target_fps (int, optional): 目标帧率，默认为10Hz
    """
    # 检查输入文件是否存在
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 打开视频文件
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {input_path}")
    
    # 获取视频属性
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算降采样因子
    if original_fps <= 0:
        raise ValueError(f"无效的原始帧率: {original_fps}")
    downsample_factor = int(round(original_fps / target_fps))
    if downsample_factor <= 0:
        raise ValueError(f"计算的降采样因子无效: {downsample_factor}")
    
    # 创建视频写入器
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 默认使用MPEG-4编码
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        target_fps,
        (width, height),
        isColor=True
    )
    
    if not out.isOpened():
        # 尝试其他编码器
        alternative_codecs = ['avc1', 'x264', 'XVID']
        for codec in alternative_codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                target_fps,
                (width, height),
                isColor=True
            )
            if out.isOpened():
                break
        else:
            raise RuntimeError(f"无法创建视频写入器，尝试了多种编码器")
    
    # 处理进度跟踪
    progress_bar = tqdm(total=total_frames, desc=f"处理 {input_path.name}", unit="帧")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        progress_bar.update(1)
        
        # 根据降采样因子选择帧
        if frame_count % downsample_factor == 0:
            # 确保图像格式正确
            if len(frame.shape) == 2:  # 灰度图转BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
    
    # 释放资源
    cap.release()
    out.release()
    progress_bar.close()
    
    logging.info(f"视频降采样完成: {input_path} -> {output_path} (帧率: {original_fps} -> {target_fps})")
    
def downsample_dataset(src_root, dst_root):
    """主处理函数"""
    # 检查并创建目录
    dst_path = check_and_create_dirs(src_root, dst_root)
    
    # 处理元数据
    update_metadata(Path(src_root)/"meta", dst_path/"meta")
    
    # 处理Parquet文件
    for parquet_path in Path(src_root).glob("data/**/*.parquet"):
        relative = parquet_path.relative_to(src_root)
        downsample_parquet(parquet_path, dst_path/relative)
    
    # 处理视频文件
    for video_path in Path(src_root).glob("videos/**/*.mp4"):
        relative = video_path.relative_to(src_root)
        downsample_video(video_path, dst_path/relative)
    
    # 重新计算统计信息
    compute_stats(dst_path/"data")

def validate_timestamps(dataset_path):
    """验证时间戳同步性"""
    for parquet_path in Path(dataset_path).glob("data/**/*.parquet"):
        df = pd.read_parquet(parquet_path)
        timestamps = df["timestamp"].values
        
        # 检查时间戳是否连续
        for i in range(1, len(timestamps)):
            diff = timestamps[i] - timestamps[i-1]
            if abs(diff - (1 / 10)) > 0.1:  # 10Hz的预期时间间隔为0.1秒
                print(f"时间戳不连续: {parquet_path} (差异: {diff})")
                break
            
if __name__ == "__main__":
    src_path = "/home/kwj/GitCode/lerobot/data/pull"
    dst_path = "/home/kwj/GitCode/lerobot/data/pull1"
    
    # 验证时间戳
    # validate_timestamps(dst_path)
    # test_df = pd.read_parquet("/home/kwj/GitCode/lerobot/data/test/data/chunk-000/episode_000000.parquet")
    # print("Columns in original data:", test_df.columns.tolist())
    try:
        downsample_dataset(src_path, dst_path)
        print("降采样完成")
    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise