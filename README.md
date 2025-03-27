# Jeppesen航图地理校准模型

这个项目使用深度学习方法将Jeppesen航图（PNG格式）与地理坐标对应，使航图能够在地理信息系统中正确显示。

## 功能

- 自动识别航图中的关键点并与地理坐标对应
- 使用深度学习模型预测航图的地理变换矩阵
- 支持手动校准航图
- 批量处理多张航图
- 可视化校准结果

## 依赖

- Python 3.7+
- PyTorch 1.8+
- OpenCV 4.5+
- 其他依赖请见 `requirements.txt`

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/jeppesen-chart-geo-calibration.git
cd jeppesen-chart-geo-calibration

# 安装依赖
pip install -r requirements.txt

# 如果需要使用文字识别功能，还需要安装Tesseract OCR
# Ubuntu/Debian:
# sudo apt-get install tesseract-ocr
# macOS:
# brew install tesseract
# Windows:
# 下载安装程序: https://github.com/UB-Mannheim/tesseract/wiki
```

## 使用方法

### 训练模型

```bash
python main.py train --config config/default_config.json
```

### 评估模型

```bash
python main.py evaluate --config config/default_config.json --model checkpoints/checkpoint_best.pth
```

### 校准单张航图

```bash
python main.py calibrate --config config/default_config.json --chart path/to/chart.png --output output/calibrated.png
```

### 手动校准航图

```bash
python main.py manual --config config/default_config.json --chart path/to/chart.png
```

### 批量校准多张航图

```bash
python main.py batch --config config/default_config.json --dir path/to/charts/
```

## 配置文件说明

配置文件使用JSON格式，主要参数包括：

- `backbone`: 使用的骨干网络，可选 "resnet50"、"efficientnet_b3" 等
- `num_control_points`: 控制点数量
- `transformation_type`: 变换类型，可选 "affine" 或 "perspective"
- `train_airport`, `val_airport`, `test_airport`: 训练、验证和测试的机场ICAO代码
- `image_size`: 图像处理的大小
- 更多参数见默认配置文件 `config/default_config.json`

## 数据库

项目使用SQLite数据库存储航空数据，包括：

- 机场信息
- 跑道信息
- 导航台信息
- 航路点信息
- 航图信息

## 校准结果

校准结果会以JSON格式保存，包含：

- 变换矩阵
- 控制点坐标
- 准确度信息
- 机场ICAO代码（如果有）

## 可视化

校准结果会生成可视化图像，显示：

- 控制点位置
- 机场位置
- 跑道位置
- 导航台位置

## 注意事项

- 确保航图图像清晰可见
- 最好使用带有明显参考点（如跑道、导航台）的航图
- 对于没有明显参考点的航图，推荐使用手动校准方式

## 协议

GPL协议
