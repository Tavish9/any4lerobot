# Libero统一数据转换器

## 概述

这个统一转换器支持自动识别和转换两种Libero数据格式：
- **RLDS格式**: TensorFlow Records格式的Libero数据集
- **HDF5格式**: 基于HDF5文件的自定义数据集

转换器会自动检测数据格式，并使用相应的处理器将数据转换为LeRobot格式。

## 功能特性

✅ **格式自动检测**: 自动识别RLDS和HDF5格式  
✅ **多线程处理**: 支持并行处理提高转换速度  
✅ **视频压缩**: 支持视频格式存储图像数据，节省60倍存储空间  
✅ **灵活配置**: 支持自定义特征配置文件  
✅ **Hub集成**: 直接推送到Hugging Face Hub  

## 安装依赖

```bash
# 基础依赖
pip install opencv-python h5py tqdm numpy

# RLDS支持（可选）
pip install tensorflow tensorflow-datasets

# LeRobot核心包
pip install lerobot
```

## 使用方法

### 1. 基本用法（自动格式检测）

```bash
python run_converter.py \
  --data-dir /path/to/your/data \
  --repo-id username/dataset_name \
  --use-videos \
  --push-to-hub
```

### 2. HDF5格式数据

```bash
# 使用默认配置
python run_converter.py \
  --data-dir /path/to/hdf5/data \
  --repo-id username/hdf5_dataset \
  --task-name "pick_and_place" \
  --num-workers 8

# 使用自定义配置文件
python run_converter.py \
  --data-dir /path/to/hdf5/data \
  --repo-id username/hdf5_dataset \
  --config config_example.json \
  --task-name "manipulation_task" \
  --num-workers 4
```

### 3. RLDS格式数据（传统模式）

```bash
python run_converter.py \
  --data-dir /path/to/rlds/data \
  --repo-id username/libero_dataset \
  --use-videos \
  --fps 20
```

### 4. 强制指定格式

```bash
# 强制使用HDF5处理器
python run_converter.py \
  --data-dir /path/to/data \
  --repo-id username/dataset \
  --force-format hdf5 \
  --task-name "custom_task"

# 强制使用RLDS处理器
python run_converter.py \
  --data-dir /path/to/data \
  --repo-id username/dataset \
  --force-format rlds
```

## 数据格式要求

### HDF5格式

期望的目录结构：
```
data_dir/
├── episode_001/
│   └── data/
│       └── trajectory.hdf5
├── episode_002/
│   └── data/
│       └── trajectory.hdf5
└── ...
```

或者直接包含HDF5文件：
```
data_dir/
├── trajectory_001.hdf5
├── trajectory_002.hdf5
└── ...
```

HDF5文件内部结构：
```
trajectory.hdf5
├── puppet/
│   ├── joint_position      # 机器人关节位置 [N, 7]
│   └── ...
├── observations/
│   ├── rgb_images/
│   │   └── camera_top     # RGB图像数据 [N, H, W, 3]
│   └── ...
```

### RLDS格式

支持的数据集：
- `libero_10_no_noops`
- `libero_goal_no_noops`
- `libero_object_no_noops`
- `libero_spatial_no_noops`

数据集应包含标准的TensorFlow Records文件和dataset_info.json。

## 配置文件格式

对于HDF5数据，可以创建JSON配置文件来定义特征结构：

```json
{
  "observation.images.camera_top": {
    "dtype": "image",
    "shape": [640, 360, 3],
    "names": ["height", "width", "channel"]
  },
  "observation.state": {
    "dtype": "float32",
    "shape": [7],
    "names": ["state_0", "state_1", "state_2", "state_3", "state_4", "state_5", "state_6"]
  },
  "action": {
    "dtype": "float32", 
    "shape": [7],
    "names": ["action_0", "action_1", "action_2", "action_3", "action_4", "action_5", "action_6"]
  }
}
```
需要查看数据格式，可以使用下面的方法
```bash
h5dump -H /media/bigdisk/Isaac-GR00T/demo_data/libero_hdf5/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 | head -300
```
或者使用python代码

```python
import h5py
import numpy as np

file_path = '/media/bigdisk/Isaac-GR00T/demo_data/libero_hdf5/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5'

def print_hdf5_structure(file_path, max_depth=3):
    def print_keys(obj, level=0, max_depth=3):
        if level > max_depth:
            return
        
        if hasattr(obj, 'keys'):
            for key in obj.keys():
                print('  ' * level + f'{key}')
                if hasattr(obj[key], 'keys'):
                    print_keys(obj[key], level+1, max_depth)
                else:
                    # 打印数据集的形状信息
                    try:
                        shape = obj[key].shape
                        dtype = obj[key].dtype
                        print('  ' * (level+1) + f'-> shape: {shape}, dtype: {dtype}')
                    except:
                        pass
    
    with h5py.File(file_path, 'r') as f:
        print(f'HDF5文件结构: {file_path}')
        print_keys(f, max_depth=max_depth)

print_hdf5_structure(file_path)
```

## 命令行参数

### 必需参数
- `--data-dir`: 数据目录路径
- `--repo-id`: 数据集仓库ID (格式: username/dataset_name)

### 输出配置
- `--output-dir`: 本地输出目录
- `--push-to-hub`: 推送到Hugging Face Hub
- `--private`: 创建私有数据集

### 数据格式
- `--use-videos`: 使用视频格式存储图像
- `--robot-type`: 机器人类型 (默认: panda)
- `--fps`: 帧率 (默认: 20)

### HDF5特定
- `--config`: 配置文件路径
- `--task-name`: 任务名称

### 性能调优
- `--num-workers`: 并行线程数 (默认: 4)
- `--image-writer-processes`: 图像写入进程数 (默认: 5)
- `--image-writer-threads`: 图像写入线程数 (默认: 10)

### 调试选项
- `--verbose`: 详细日志
- `--dry-run`: 试运行模式
- `--force-format`: 强制指定格式 (rlds/hdf5)

## 性能优化建议

1. **多线程处理**: 增加`--num-workers`可以提高处理速度，但要注意内存使用
2. **视频压缩**: 使用`--use-videos`可以显著减少存储空间
3. **图像写入**: 调整`--image-writer-processes`和`--image-writer-threads`参数
4. **内存管理**: 处理大数据集时，考虑分批处理

## 故障排除

### 常见错误

1. **导入错误**
   ```
   ImportError: No module named 'cv2'
   ```
   解决：`pip install opencv-python`

2. **HDF5文件找不到**
   ```
   ValueError: 无法检测数据格式
   ```
   解决：检查数据目录结构，确保包含.hdf5文件

3. **内存不足**
   ```
   MemoryError: 
   ```
   解决：减少`--num-workers`参数，或分批处理数据

### 调试模式

使用`--verbose --dry-run`可以检查配置而不执行转换：

```bash
python run_converter.py \
  --data-dir /path/to/data \
  --repo-id test/dataset \
  --verbose \
  --dry-run
```

## 示例工作流

### 完整的HDF5转换流程

```bash
# 1. 检查数据结构
python run_converter.py \
  --data-dir ./hdf5_data \
  --repo-id test/check \
  --dry-run --verbose

# 2. 本地转换测试
python run_converter.py \
  --data-dir ./hdf5_data \
  --repo-id test/local_test \
  --task-name "manipulation" \
  --num-workers 4

# 3. 推送到Hub
python run_converter.py \
  --data-dir ./hdf5_data \
  --repo-id username/final_dataset \
  --task-name "manipulation" \
  --push-to-hub \
  --tags manipulation robotics panda \
  --num-workers 8
```

## 贡献

欢迎提交Issue和Pull Request来改进这个转换器！

## 许可证

Apache 2.0 