# FY-4B卫星火点数据处理工具

本工具用于处理FY-4B卫星数据，实现火点检测、坐标转换、数据可视化等功能。

## 功能特点

1. **火点检测**：基于3.75μm通道数据，结合云检测和温度对比进行火点识别
2. **坐标转换**：支持经纬度与行列号的双向转换
3. **数据可视化**：提供火点检测结果和地理分布的可视化展示
4. **批量处理**：支持批量处理多个HDF文件
5. **参数可调**：提供灵活的参数设置接口，适应不同场景需求

## 文件结构

- `FY4B_fire_data_processor.py` - 核心处理类
- `test_FY4B_fire_processor.py` - 测试脚本
- `output_fire_data/` - 默认输出目录（自动创建）

## 安装依赖

```bash
pip install numpy h5py pandas matplotlib cartopy scipy
```

## 使用方法

### 方法一：使用测试脚本（推荐）

运行测试脚本，选择相应的功能：

```bash
python test_FY4B_fire_processor.py
```

测试脚本提供以下功能：
1. 处理单个HDF文件
2. 批量处理目录中的所有HDF文件
3. 测试坐标转换功能

### 方法二：在Python代码中调用

```python
from FY4B_fire_data_processor import FY4BFireDataProcessor

# 创建处理器实例
processor = FY4BFireDataProcessor()

# 调整参数
processor.hot_threshold_low = 310  # 火点温度阈值（低）
processor.hot_threshold_high = 330  # 高温火点阈值
processor.cloud_threshold = 0.6  # 云量阈值
processor.min_hot_contrast = 10  # 对比度要求
processor.min_hot_area = 1  # 最小火点面积（像素）

# 处理单个文件
fire_points = processor.process_single_file("path/to/file.HDF", output_dir="./output", visualize=True)

# 批量处理
processor.batch_process("path/to/data_dir", output_dir="./output_batch", visualize=True)
```

## 主要参数说明

| 参数 | 说明 | 默认值 | 调整建议 |
|------|------|--------|----------|
| hot_threshold_low | 火点检测温度阈值（K） | 310 | 降低可检测更多潜在火点，但可能增加误判；升高可减少误判，但可能漏检弱火点 |
| hot_threshold_high | 高温火点温度阈值（K） | 330 | 用于区分一般火点和高温火点 |
| cloud_threshold | 云量阈值 | 0.6 | 高于此值的区域被视为云区，0-1之间 |
| min_hot_contrast | 火点与背景的最小温度差（K） | 10 | 降低可检测对比度较低的火点，但可能增加误判 |
| min_hot_area | 最小火点面积（像素） | 1 | 增加可过滤小面积噪声，但可能漏检小面积火点 |
| context_window | 背景窗口大小 | 5 | 用于计算火点周围背景温度的窗口大小 |

## 输出结果

1. **CSV文件**：
   - 每个文件的火点检测结果（`文件名_fire_points.csv`）
   - 批量处理时的汇总文件（`all_fire_points_summary.csv`）

2. **可视化图像**：
   - 火点检测结果图（`文件名_fire_detection.png`）
   - 地理分布可视化图（`文件名_geo_fire.png`）

## 数据格式说明

输出的CSV文件包含以下字段：
- `lat`: 纬度
- `lon`: 经度
- `temperature`: 温度（K）
- `strength`: 强度（1: 一般火点，2: 高温火点）

## 注意事项

1. 请确保安装了所有依赖库
2. 处理大量数据时可能需要较长时间和较大内存
3. 建议先使用单个文件测试，调整参数后再进行批量处理
4. 坐标转换可能存在一定误差，特别是在边缘区域
5. 云检测算法在复杂天气条件下可能存在误判

## 参数调优指南

### 增加检测灵敏度

如果需要检测更多潜在火点（可能增加误判）：
- 降低 `hot_threshold_low`（如290-300K）
- 降低 `min_hot_contrast`（如5-8K）
- 提高 `cloud_threshold`（如0.7-0.8）
- 保持 `min_hot_area` 为1

### 减少误判

如果需要减少误判（可能增加漏检）：
- 提高 `hot_threshold_low`（如315-320K）
- 提高 `min_hot_contrast`（如12-15K）
- 降低 `cloud_threshold`（如0.5）
- 增加 `min_hot_area`（如2-3像素）

### 针对不同区域的调整

- **森林区域**：可适当提高灵敏度，如降低温度阈值
- **城市区域**：应提高阈值减少城市热岛效应干扰
- **干旱区域**：需注意云和沙尘的影响，可能需要调整云检测参数

## 故障排除

1. **文件无法读取**：检查文件路径和权限，确保文件格式正确
2. **坐标转换错误**：检查分辨率设置是否正确
3. **内存不足**：处理大文件时可能需要增加内存或分块处理
4. **可视化失败**：确保matplotlib和cartopy正确安装
5. **无火点检测结果**：尝试降低温度阈值和对比度要求

## 扩展开发

如需扩展功能，可以考虑：
1. 添加时间窗口验证功能，提高火点检测准确性
2. 实现多源数据融合
3. 增加更复杂的云检测算法
4. 开发实时处理功能

## 版本信息

- 版本：1.0.0
- 开发日期：2024
- 支持数据格式：FY-4B HDF格式

## 联系方式

如有问题或建议，请联系开发者。