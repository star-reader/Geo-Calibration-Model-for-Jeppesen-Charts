import argparse
import logging
import sys
import os
import torch
import numpy as np
import random
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import cv2
import datetime

from database.aviation_db import AviationDatabase
from preprocessing.data_loader import get_chart_dataloader, ChartPreprocessor
from model.chart_calibration_model import create_chart_calibration_model, create_chart_calibration_loss
from training.trainer import ChartCalibrationTrainer
from utils.geo_utils import (
    create_transformation_matrix, 
    apply_transformation, 
    pixel_to_geo, 
    geo_to_pixel,
    calculate_transformation_accuracy,
    optimize_transformation
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jeppesen_geo_calibration.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"随机种子已设置为: {seed}")

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def train(config: Dict):
    logger.info("开始模型训练...")
    set_seed(config.get('seed', 42))
    # 准备数据集
    db = AviationDatabase(config.get('db_path', 'database/aviation.db'))

    logger.info("加载训练和验证数据...")
    train_dataloader = get_chart_dataloader(
        db=db,
        chart_paths=config.get('train_charts'),
        airport_icao=config.get('train_airport'),
        chart_type=config.get('train_chart_type'),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        target_size=tuple(config.get('image_size', (1024, 1024)))
    )
    
    val_dataloader = None
    if config.get('val_charts') or config.get('val_airport'):
        val_dataloader = get_chart_dataloader(
            db=db,
            chart_paths=config.get('val_charts'),
            airport_icao=config.get('val_airport'),
            chart_type=config.get('val_chart_type'),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            target_size=tuple(config.get('image_size', (1024, 1024)))
        )
    
    logger.info(f"创建模型，使用{config['backbone']}作为骨干网络...")
    model = create_chart_calibration_model(
        backbone=config['backbone'],
        pretrained=config.get('pretrained', True),
        num_control_points=config.get('num_control_points', 4),
        transformation_type=config.get('transformation_type', 'perspective')
    )
    
    # 损失函数
    loss_fn = create_chart_calibration_loss(
        point_weight=config.get('point_weight', 1.0),
        transform_weight=config.get('transform_weight', 1.0),
        consistency_weight=config.get('consistency_weight', 0.5),
        smoothness_weight=config.get('smoothness_weight', 0.1)
    )
    
    # 训练器
    trainer = ChartCalibrationTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=loss_fn,
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        config=config
    )
    
    if config.get('resume_from'):
        trainer.load_checkpoint(config['resume_from'])
    
    best_model_path = trainer.train()
    logger.info(f"训练完成。最佳模型保存在: {best_model_path}")
    return best_model_path

def evaluate(config: Dict, model_path: str = None):
    logger.info("开始模型评估...")

    db = AviationDatabase(config.get('db_path', 'database/aviation.db'))
    logger.info("加载测试数据...")
    test_dataloader = get_chart_dataloader(
        db=db,
        chart_paths=config.get('test_charts'),
        airport_icao=config.get('test_airport'),
        chart_type=config.get('test_chart_type'),
        batch_size=1,  # 评估时使用批次大小为1
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        target_size=tuple(config.get('image_size', (1024, 1024)))
    )
    
    logger.info(f"创建模型，使用{config['backbone']}作为骨干网络...")
    model = create_chart_calibration_model(
        backbone=config['backbone'],
        pretrained=False,  # 评估时不需要预训练权重
        num_control_points=config.get('num_control_points', 4),
        transformation_type=config.get('transformation_type', 'perspective')
    )
    loss_fn = create_chart_calibration_loss(
        point_weight=config.get('point_weight', 1.0),
        transform_weight=config.get('transform_weight', 1.0),
        consistency_weight=config.get('consistency_weight', 0.5),
        smoothness_weight=config.get('smoothness_weight', 0.1)
    )
    trainer = ChartCalibrationTrainer(
        model=model,
        train_dataloader=None,
        val_dataloader=None,
        criterion=loss_fn,
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        config=config
    )
    model_path = model_path or config.get('eval_model')
    if not model_path:
        raise ValueError("必须提供模型路径进行评估")
    
    logger.info(f"加载模型: {model_path}")
    trainer.load_checkpoint(model_path)
    metrics = trainer.evaluate(test_dataloader)
    logger.info("评估结果:")
    for k, v in metrics.items():
        logger.info(f"- {k}: {v:.4f}")
    
    return metrics

def calibrate_chart(config: Dict, chart_path: str, output_path: str = None, model_path: str = None):
    logger.info(f"校准航图: {chart_path}")
    db = AviationDatabase(config.get('db_path', 'database/aviation.db'))
    preprocessor = ChartPreprocessor({
        'target_size': tuple(config.get('image_size', (1024, 1024)))
    })
    
    image = preprocessor.load_image(chart_path)
    image_processed = preprocessor.preprocess_chart(image)
    
    model = create_chart_calibration_model(
        backbone=config['backbone'],
        pretrained=False,
        num_control_points=config.get('num_control_points', 4),
        transformation_type=config.get('transformation_type', 'perspective')
    )
    
    model_path = model_path or config.get('calibration_model')
    if not model_path:
        raise ValueError("必须提供模型路径进行校准")
    
    logger.info(f"加载模型: {model_path}")
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 转换为张量
    input_tensor = torch.from_numpy(image_processed).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
    control_points = outputs['control_points'][0].cpu().numpy()
    transform_params = outputs['transform_params'][0].cpu().numpy()
    if model.transformation_type == 'affine':
        matrix = np.eye(3)
        matrix[:2, :] = transform_params.reshape(2, 3)
    else:  # perspective
        matrix = np.ones((3, 3))
        matrix.flat[:8] = transform_params
    image_size = config.get('image_size', (1024, 1024))
    pixel_points = control_points * np.array([image_size[0], image_size[1]])
    
    logger.info("预测的控制点（像素坐标）:")
    for i, point in enumerate(pixel_points):
        logger.info(f"点 {i+1}: ({point[0]:.2f}, {point[1]:.2f})")
    
    logger.info(f"预测的变换矩阵:\n{matrix}")
    airport_icao = None
    chart_name = Path(chart_path).stem
    parts = chart_name.split('_')
    if len(parts) >= 1 and len(parts[0]) == 4:
        possible_icao = parts[0]
        # 验证
        airport = db.get_airport(possible_icao)
        if not airport.empty:
            airport_icao = possible_icao
    if airport_icao:
        airport = db.get_airport(airport_icao).iloc[0]
        airport_lat, airport_lon = airport['latitude'], airport['longitude']
        
        logger.info(f"机场: {airport_icao} ({airport['name']})")
        logger.info(f"机场位置: 纬度 {airport_lat}, 经度 {airport_lon}")
        
        airport_px, airport_py = geo_to_pixel(airport_lat, airport_lon, matrix)
        logger.info(f"机场在图像中的位置（像素）: x={airport_px:.2f}, y={airport_py:.2f}")
        
        runways = db.get_runways(airport_icao)
        if not runways.empty:
            for i, runway in runways.iterrows():
                # 起点
                start_lat, start_lon = runway['latitude_start'], runway['longitude_start']
                start_px, start_py = geo_to_pixel(start_lat, start_lon, matrix)
                
                # 终点
                end_lat, end_lon = runway['latitude_end'], runway['longitude_end']
                end_px, end_py = geo_to_pixel(end_lat, end_lon, matrix)
    
    if output_path or config.get('visualize', True):
        visualize_calibration(
            image, 
            pixel_points, 
            matrix, 
            airport_icao, 
            db, 
            output_path or f"calibrated_{chart_name}.png"
        )
    
    if config.get('save_results', True):
        output_dir = Path(config.get('output_dir', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            'chart_path': chart_path,
            'transformation_matrix': matrix.tolist(),
            'control_points': pixel_points.tolist(),
            'airport_icao': airport_icao
        }
        
        result_path = output_dir / f"{chart_name}_calibration.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"校准结果已保存到: {result_path}")
    
    return {
        'control_points': pixel_points,
        'transformation_matrix': matrix,
        'airport_icao': airport_icao
    }

def visualize_calibration(image: np.ndarray, control_points: np.ndarray, 
                        matrix: np.ndarray, airport_icao: str, 
                        db: AviationDatabase, output_path: str):
    viz_img = image.copy()
    for i, (x, y) in enumerate(control_points):
        cv2.circle(viz_img, (int(x), int(y)), 10, (255, 0, 0), -1)
        cv2.putText(viz_img, f"Point {i+1}", (int(x) + 15, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    if airport_icao:
        airport = db.get_airport(airport_icao)
        if not airport.empty:
            airport = airport.iloc[0]
            airport_lat, airport_lon = airport['latitude'], airport['longitude']
            airport_px, airport_py = geo_to_pixel(airport_lat, airport_lon, matrix)
            
            if 0 <= airport_px < image.shape[1] and 0 <= airport_py < image.shape[0]:
                cv2.circle(viz_img, (int(airport_px), int(airport_py)), 15, (0, 255, 0), -1)
                cv2.putText(viz_img, f"{airport_icao}", (int(airport_px) + 20, int(airport_py)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            runways = db.get_runways(airport_icao)
            for i, runway in runways.iterrows():
                start_lat, start_lon = runway['latitude_start'], runway['longitude_start']
                start_px, start_py = geo_to_pixel(start_lat, start_lon, matrix)
                end_lat, end_lon = runway['latitude_end'], runway['longitude_end']
                end_px, end_py = geo_to_pixel(end_lat, end_lon, matrix)
                
                # 确保在图像范围内
                if (0 <= start_px < image.shape[1] and 0 <= start_py < image.shape[0] and
                    0 <= end_px < image.shape[1] and 0 <= end_py < image.shape[0]):
                    cv2.line(viz_img, (int(start_px), int(start_py)), 
                            (int(end_px), int(end_py)), (0, 0, 255), 3)
                    middle_px = (start_px + end_px) / 2
                    middle_py = (start_py + end_py) / 2
                    cv2.putText(viz_img, runway['designation'], (int(middle_px), int(middle_py)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            navaids = db.get_navaids_near_airport(airport_icao, radius_nm=20)
            for i, navaid in navaids.iterrows():
                nav_lat, nav_lon = navaid['latitude'], navaid['longitude']
                nav_px, nav_py = geo_to_pixel(nav_lat, nav_lon, matrix)
                if 0 <= nav_px < image.shape[1] and 0 <= nav_py < image.shape[0]:
                    cv2.circle(viz_img, (int(nav_px), int(nav_py)), 8, (255, 255, 0), -1)
                    cv2.putText(viz_img, navaid['ident'], (int(nav_px) + 10, int(nav_py)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    plt.figure(figsize=(16, 16))
    plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
    plt.title(f"航图校准结果 - {Path(output_path).stem}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"可视化结果已保存到: {output_path}")

def manual_calibration(config: Dict, chart_path: str, output_path: str = None):
    logger.info(f"开始手动校准航图: {chart_path}")
    
    db = AviationDatabase(config.get('db_path', 'database/aviation.db'))
    image = cv2.imread(chart_path)
    if image is None:
        raise FileNotFoundError(f"找不到图像或无法加载: {chart_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    airport_icao = None
    chart_name = Path(chart_path).stem
    
    parts = chart_name.split('_')
    if len(parts) >= 1 and len(parts[0]) == 4:
        possible_icao = parts[0]
        airport = db.get_airport(possible_icao)
        if not airport.empty:
            airport_icao = possible_icao
    
    reference_points = []
    geo_points = []
    
    if airport_icao:
        # 显示机场相关信息
        airport = db.get_airport(airport_icao).iloc[0]
        
        # 显示跑道信息
        runways = db.get_runways(airport_icao)
        if not runways.empty:
            print("\n可用跑道:")
            for i, runway in runways.iterrows():
                print(f"{i+1}. {runway['designation']}: 起点({runway['latitude_start']}, {runway['longitude_start']}), "
                     f"终点({runway['latitude_end']}, {runway['longitude_end']})")
        
        navaids = db.get_navaids_near_airport(airport_icao, radius_nm=20)
        if not navaids.empty:
            print("\n附近导航台:")
            for i, navaid in navaids.iterrows():
                print(f"{i+1}. {navaid['ident']} ({navaid['type']}): ({navaid['latitude']}, {navaid['longitude']})")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_rgb)
    
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        if ix is not None and iy is not None:
            ax.plot(ix, iy, 'ro', markersize=8)
            ax.text(ix + 10, iy, f"点 {len(reference_points)+1}", color='red', fontsize=12)
            fig.canvas.draw()
            
            reference_points.append([ix, iy])
            
            lat = float(input(f"点 {len(reference_points)} 的纬度（度）: "))
            lon = float(input(f"点 {len(reference_points)} 的经度（度）: "))
            geo_points.append([lon, lat])  # 注意顺序是 [经度, 纬度]
            
            print(f"已添加点 {len(reference_points)}: 像素({ix:.2f}, {iy:.2f}) -> 地理({lat}, {lon})")
            
            if len(reference_points) >= 4:
                plt.close(fig)
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    # 确保至少有4个点
    if len(reference_points) < 4:
        raise ValueError("需要至少4个参考点来进行校准")
    
    logger.info(f"已收集{len(reference_points)}个参考点")
    reference_points = np.array(reference_points)
    geo_points = np.array(geo_points)
    transform_type = config.get('transformation_type', 'perspective')
    matrix = create_transformation_matrix(reference_points, geo_points, transform_type)
    optimized_matrix = optimize_transformation(
        reference_points, geo_points, matrix, transform_type
    )
    accuracy = calculate_transformation_accuracy(
        reference_points, geo_points, optimized_matrix
    )
    
    logger.info(f"变换矩阵:\n{optimized_matrix}")
    logger.info(f"校准精度: RMSE = {accuracy['rmse']:.6f}, MAE = {accuracy['mae']:.6f}")
    
    visualize_calibration(
        image_rgb, 
        reference_points, 
        optimized_matrix, 
        airport_icao, 
        db, 
        output_path or f"manual_calibrated_{chart_name}.png"
    )
    
    if config.get('save_results', True):
        output_dir = Path(config.get('output_dir', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            'chart_path': chart_path,
            'transformation_matrix': optimized_matrix.tolist(),
            'reference_points': reference_points.tolist(),
            'geo_points': geo_points.tolist(),
            'airport_icao': airport_icao,
            'accuracy': accuracy
        }
        
        result_path = output_dir / f"{chart_name}_manual_calibration.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"手动校准结果已保存到: {result_path}")
    
    return {
        'reference_points': reference_points,
        'transformation_matrix': optimized_matrix,
        'airport_icao': airport_icao,
        'accuracy': accuracy
    }

def batch_calibrate(config: Dict, chart_dir: str, output_dir: str = None, model_path: str = None):
    logger.info(f"开始批量校准航图: {chart_dir}")
    
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path(config.get('output_dir', 'output'))
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    chart_path = Path(chart_dir)
    image_extensions = ['.png', '.jpg']
    chart_files = []
    
    for ext in image_extensions:
        chart_files.extend(chart_path.glob(f"*{ext}"))
    
    logger.info(f"找到{len(chart_files)}个图像文件")
    
    # 校准每张图像
    results = []
    for chart_file in chart_files:
        try:
            logger.info(f"处理: {chart_file}")
            output_file = output_path / f"calibrated_{chart_file.name}"
            
            result = calibrate_chart(
                config, 
                str(chart_file), 
                str(output_file),
                model_path
            )
            
            results.append({
                'chart_path': str(chart_file),
                'output_path': str(output_file),
                'success': True,
                'control_points': result['control_points'].tolist() if isinstance(result['control_points'], np.ndarray) else result['control_points'],
                'transformation_matrix': result['transformation_matrix'].tolist() if isinstance(result['transformation_matrix'], np.ndarray) else result['transformation_matrix'],
                'airport_icao': result['airport_icao']
            })
            
        except Exception as e:
            logger.error(f"处理{chart_file}时出错: {e}")
            results.append({
                'chart_path': str(chart_file),
                'success': False,
                'error': str(e)
            })
    
    summary_path = output_path / "batch_calibration_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.datetime.now().isoformat(),
            'total_charts': len(chart_files),
            'success_count': sum(1 for r in results if r.get('success', False)),
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"批量校准完成。成功: {sum(1 for r in results if r.get('success', False))}/{len(chart_files)}")
    logger.info(f"摘要已保存到: {summary_path}")

def main():
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', required=True, help='配置文件路径')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--config', required=True, help='配置文件路径')
    eval_parser.add_argument('--model', help='模型路径（可选，如不提供则使用配置中的路径）')
    
    # 校准命令
    calibrate_parser = subparsers.add_parser('calibrate', help='校准单张航图')
    calibrate_parser.add_argument('--config', required=True, help='配置文件路径')
    calibrate_parser.add_argument('--chart', required=True, help='航图路径')
    calibrate_parser.add_argument('--output', help='输出文件路径')
    calibrate_parser.add_argument('--model', help='模型路径（可选，如不提供则使用配置中的路径）')
    
    # 手动校准命令
    manual_parser = subparsers.add_parser('manual', help='手动校准航图')
    manual_parser.add_argument('--config', required=True, help='配置文件路径')
    manual_parser.add_argument('--chart', required=True, help='航图路径')
    manual_parser.add_argument('--output', help='输出文件路径')
    
    # 批量校准命令
    batch_parser = subparsers.add_parser('batch', help='批量校准多张航图')
    batch_parser.add_argument('--config', required=True, help='配置文件路径')
    batch_parser.add_argument('--dir', required=True, help='航图目录')
    batch_parser.add_argument('--output', help='输出目录')
    batch_parser.add_argument('--model', help='模型路径（可选，如不提供则使用配置中的路径）')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        config = load_config(args.config)
        
        if args.command == 'train':
            train(config)
        
        elif args.command == 'evaluate':
            evaluate(config, args.model)
        
        elif args.command == 'calibrate':
            calibrate_chart(config, args.chart, args.output, args.model)
        
        elif args.command == 'manual':
            manual_calibration(config, args.chart, args.output)
        
        elif args.command == 'batch':
            batch_calibrate(config, args.dir, args.output, args.model)
    
    except Exception as e:
        logger.error(f"执行{args.command}命令时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
