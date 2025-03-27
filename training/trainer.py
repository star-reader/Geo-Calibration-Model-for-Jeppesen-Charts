import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import time
import random

from model.chart_calibration_model import ChartCalibrationModel, ChartCalibrationLoss
from database.aviation_db import AviationDatabase
from preprocessing.data_loader import ChartDataset
from utils.geo_utils import create_transformation_matrix, calculate_transformation_accuracy

logger = logging.getLogger(__name__)

class ChartCalibrationTrainer:
    """航图校准模型训练器"""
    
    def __init__(self, 
                 model: ChartCalibrationModel,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader = None,
                 criterion: nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 device: str = 'cuda',
                 config: Dict = None):
        """
        初始化训练器
        
        参数:
            model: 航图校准模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 计算设备
            config: 配置字典
        """
        self.config = config or {}
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # 设置损失函数
        self.criterion = criterion or ChartCalibrationLoss(
            point_weight=self.config.get('point_weight', 1.0),
            transform_weight=self.config.get('transform_weight', 1.0),
            consistency_weight=self.config.get('consistency_weight', 0.5),
            smoothness_weight=self.config.get('smoothness_weight', 0.1)
        )
        
        # 设置优化器
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # 设置学习率调度器
        if scheduler is None:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = scheduler
            
        # 设置设备
        self.device = device
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA不可用，使用CPU进行训练")
            self.device = 'cpu'
        
        self.model = self.model.to(self.device)
        
        # 训练配置
        self.num_epochs = self.config.get('num_epochs', 50)
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置TensorBoard
        self.tensorboard_dir = Path(self.config.get('tensorboard_dir', 'runs'))
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # 数据库实例（用于保存校准结果）
        self.db = AviationDatabase()
        
        logger.info("训练器初始化完成")
    
    def train_one_epoch(self) -> Dict[str, float]:
        """
        训练一个Epoch
        
        返回:
            包含训练指标的字典
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {
            'point_loss': 0.0,
            'transform_loss': 0.0,
            'consistency_loss': 0.0,
            'smoothness_loss': 0.0
        }
        
        num_batches = len(self.train_dataloader)
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 准备输入和目标
            images = batch['image'].to(self.device)
            metadata = batch['metadata']
            
            # 从元数据中提取标注（如果有）
            targets = self._extract_targets_from_metadata(metadata)
            if targets is None:
                continue  # 跳过没有标注的样本
                
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in targets.items()}
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(images)
            
            # 添加模型引用到预测中，用于一致性损失计算
            outputs['model'] = self.model
            
            # 计算损失
            loss_dict = self.criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'point_loss': f"{loss_dict['point_loss'].item():.4f}"
            })
            
            # 累加损失
            epoch_loss += loss.item()
            for k, v in loss_dict.items():
                if k in epoch_metrics:
                    epoch_metrics[k] += v.item()
            
            # 记录到TensorBoard
            global_step = self.current_epoch * num_batches + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), global_step)
            for k, v in loss_dict.items():
                self.writer.add_scalar(f'train/{k}', v.item(), global_step)
        
        # 计算平均值
        epoch_loss /= num_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        epoch_metrics['loss'] = epoch_loss
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """
        在验证集上评估模型
        
        返回:
            包含验证指标的字典
        """
        if self.val_dataloader is None:
            return {'loss': 0.0}
        
        self.model.eval()
        val_loss = 0.0
        val_metrics = {
            'point_loss': 0.0,
            'transform_loss': 0.0,
            'consistency_loss': 0.0,
            'smoothness_loss': 0.0
        }
        
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc="Validating")
            
            for batch_idx, batch in enumerate(progress_bar):
                # 准备输入和目标
                images = batch['image'].to(self.device)
                metadata = batch['metadata']
                
                # 从元数据中提取标注
                targets = self._extract_targets_from_metadata(metadata)
                if targets is None:
                    continue
                    
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in targets.items()}
                
                # 前向传播
                outputs = self.model(images)
                
                # 添加模型引用到预测中
                outputs['model'] = self.model
                
                # 计算损失
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                # 更新进度条
                progress_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})
                
                # 累加损失
                val_loss += loss.item()
                for k, v in loss_dict.items():
                    if k in val_metrics:
                        val_metrics[k] += v.item()
                
                # 可视化一些预测结果
                if batch_idx % 10 == 0:
                    self._visualize_predictions(
                        images[0], 
                        outputs['control_points'][0], 
                        targets['control_points'][0] if 'control_points' in targets else None,
                        batch_idx
                    )
        
        # 计算平均值
        val_loss /= num_batches
        for k in val_metrics:
            val_metrics[k] /= num_batches
        
        # 更新学习率调度器
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
        
        val_metrics['loss'] = val_loss
        return val_metrics
    
    def _extract_targets_from_metadata(self, metadata: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
        """
        从元数据中提取训练目标
        
        参数:
            metadata: 批次元数据列表
            
        返回:
            目标字典，如果没有可用标注则返回None
        """
        batch_size = len(metadata)
        
        # 检查是否有校准数据
        has_calibration = all('calibration' in m for m in metadata)
        
        if has_calibration:
            # 从校准数据中提取参考点和变换矩阵
            control_points = []
            transform_params = []
            world_points = []
            
            for m in metadata:
                cal = m['calibration']
                ref_points = np.array(cal['reference_points'])
                
                # 归一化参考点坐标到[0,1]范围
                image_width = 1024  # 假设标准化尺寸
                image_height = 1024
                ref_points_norm = ref_points.copy()
                ref_points_norm[:, 0] /= image_width
                ref_points_norm[:, 1] /= image_height
                
                control_points.append(ref_points_norm)
                
                # 提取变换矩阵参数
                matrix = np.array(cal['transformation_matrix'])
                
                # 根据模型的变换类型提取参数
                if self.model.transformation_type == 'affine':
                    params = matrix[:2, :].flatten()
                else:  # perspective
                    params = matrix.flatten()[:8]  # 最后一个元素固定为1
                
                transform_params.append(params)
                
                # 提取世界坐标点（如果有）
                if 'world_points' in cal:
                    world_points.append(np.array(cal['world_points']))
            
            targets = {
                'control_points': torch.tensor(control_points, dtype=torch.float32),
                'transform_params': torch.tensor(transform_params, dtype=torch.float32)
            }
            
            if world_points:
                targets['world_points'] = torch.tensor(world_points, dtype=torch.float32)
            
            return targets
        
        # 检查是否有匹配的特征点
        has_landmarks = all('landmarks' in m for m in metadata)
        
        if has_landmarks:
            # 尝试从特征点构建训练目标
            try:
                return self._build_targets_from_landmarks(metadata)
            except Exception as e:
                logger.warning(f"从特征点构建目标失败: {e}")
                return None
        
        # 没有可用标注
        return None
    
    def _build_targets_from_landmarks(self, metadata: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
        """
        从特征点构建训练目标
        
        参数:
            metadata: 批次元数据列表
            
        返回:
            目标字典，如果不能构建则返回None
        """
        batch_size = len(metadata)
        num_control_points = self.model.num_control_points
        
        control_points = []
        geo_points = []
        
        for m in metadata:
            landmarks = m['landmarks']
            
            # 找出匹配到地理特征的点
            matched_landmarks = [lm for lm in landmarks if 'matched_feature' in lm]
            
            # 如果匹配点不够，返回None
            if len(matched_landmarks) < num_control_points:
                return None
            
            # 选择最可靠的匹配点
            selected_landmarks = sorted(
                matched_landmarks, 
                key=lambda x: x.get('response', 0) if x['type'] == 'keypoint' else x.get('confidence', 0),
                reverse=True
            )[:num_control_points]
            
            # 提取像素坐标
            points = np.array([[lm['x'], lm['y']] for lm in selected_landmarks])
            
            # 归一化到[0,1]范围
            image_width = 1024  # 假设标准化尺寸
            image_height = 1024
            points_norm = points.copy()
            points_norm[:, 0] /= image_width
            points_norm[:, 1] /= image_height
            
            control_points.append(points_norm)
            
            # 提取对应的地理坐标
            geo_coords = []
            for lm in selected_landmarks:
                feature = lm['matched_feature']
                if feature['type'] == 'runway':
                    # 对于跑道，使用中点
                    lat1, lon1 = feature['latitude_start'], feature['longitude_start']
                    lat2, lon2 = feature['latitude_end'], feature['longitude_end']
                    lat = (lat1 + lat2) / 2
                    lon = (lon1 + lon2) / 2
                elif feature['type'] == 'navaid':
                    lat, lon = feature['latitude'], feature['longitude']
                else:
                    # 其他类型，暂不支持
                    continue
                geo_coords.append([lon, lat])  # 注意顺序是 [lon, lat]
            
            geo_points.append(np.array(geo_coords))
        
        # 将所有样本转换为Tensor
        targets = {
            'control_points': torch.tensor(control_points, dtype=torch.float32),
            'world_points': torch.tensor(geo_points, dtype=torch.float32)
        }
        
        return targets
    
    def _visualize_predictions(self, 
                             image: torch.Tensor, 
                             pred_points: torch.Tensor, 
                             target_points: Optional[torch.Tensor],
                             batch_idx: int):
        """
        可视化预测的控制点
        
        参数:
            image: 输入图像
            pred_points: 预测的控制点 [num_points, 2]
            target_points: 目标控制点 [num_points, 2]（可选）
            batch_idx: 批次索引
        """
        # 转换为numpy数组
        img = image.cpu().permute(1, 2, 0).numpy()
        
        # 反归一化图像
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # 创建图像
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        
        height, width = img.shape[:2]
        
        # 绘制预测点
        pred_points_np = pred_points.cpu().numpy()
        plt.scatter(
            pred_points_np[:, 0] * width, 
            pred_points_np[:, 1] * height, 
            c='r', 
            marker='x', 
            s=100, 
            label='Predicted'
        )
        
        # 绘制目标点
        if target_points is not None:
            target_points_np = target_points.cpu().numpy()
            plt.scatter(
                target_points_np[:, 0] * width, 
                target_points_np[:, 1] * height, 
                c='g', 
                marker='o', 
                s=100, 
                label='Target'
            )
        
        plt.legend()
        plt.title('Control Points Visualization')
        
        # 保存到TensorBoard
        self.writer.add_figure(
            f'validation/control_points_{batch_idx}', 
            plt.gcf(), 
            global_step=self.current_epoch
        )
        
        plt.close()
    
    def train(self):
        """
        训练模型指定的轮数
        """
        logger.info(f"开始训练，共{self.num_epochs}个Epoch")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # 训练一个Epoch
            start_time = time.time()
            train_metrics = self.train_one_epoch()
            train_time = time.time() - start_time
            
            # 验证
            val_metrics = self.validate()
            
            # 记录指标
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('epoch/learning_rate', current_lr, epoch)
            
            # 打印Epoch统计信息
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} "
                f"- Train Loss: {train_metrics['loss']:.4f} "
                f"- Val Loss: {val_metrics['loss']:.4f} "
                f"- LR: {current_lr:.6f} "
                f"- Time: {train_time:.2f}s"
            )
            
            # 保存检查点
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_model_path = self._save_checkpoint(is_best)
                logger.info(f"保存了最佳模型: {self.best_model_path}")
            else:
                self._save_checkpoint(is_best)
        
        logger.info("训练完成")
        logger.info(f"最佳验证损失: {self.best_val_loss:.4f}，模型保存在: {self.best_model_path}")
        
        # 关闭TensorBoard writer
        self.writer.close()
        
        return self.best_model_path
    
    def _save_checkpoint(self, is_best: bool) -> str:
        """
        保存检查点
        
        参数:
            is_best: 是否是最佳模型
            
        返回:
            保存的检查点路径
        """
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / f"checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # 保存特定Epoch的检查点
        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch+1}.pth"
        torch.save(checkpoint, epoch_path)
        
        # 如果是最佳模型，额外保存一个副本
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            return str(best_path)
        
        return str(epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        参数:
            checkpoint_path: 检查点路径
        """
        logger.info(f"加载检查点: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # 更新配置
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        logger.info(f"成功加载检查点，从Epoch {self.current_epoch}继续训练")
    
    def evaluate(self, test_dataloader: DataLoader) -> Dict[str, float]:
        """
        在测试集上评估模型
        
        参数:
            test_dataloader: 测试数据加载器
            
        返回:
            包含评估指标的字典
        """
        self.model.eval()
        test_loss = 0.0
        test_metrics = {
            'point_loss': 0.0,
            'transform_loss': 0.0,
            'consistency_loss': 0.0,
            'smoothness_loss': 0.0,
            'pixel_error': 0.0,
            'geo_error': 0.0
        }
        
        num_batches = len(test_dataloader)
        all_errors = []
        
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader, desc="Testing")
            
            for batch_idx, batch in enumerate(progress_bar):
                # 准备输入和目标
                images = batch['image'].to(self.device)
                metadata = batch['metadata']
                chart_paths = batch['chart_path']
                
                # 从元数据中提取标注
                targets = self._extract_targets_from_metadata(metadata)
                if targets is None:
                    continue
                    
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in targets.items()}
                
                # 前向传播
                outputs = self.model(images)
                
                # 添加模型引用到预测中
                outputs['model'] = self.model
                
                # 计算损失
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                # 累加损失
                test_loss += loss.item()
                for k, v in loss_dict.items():
                    if k in test_metrics:
                        test_metrics[k] += v.item()
                
                # 计算像素误差
                pred_points = outputs['control_points']
                target_points = targets['control_points']
                pixel_error = torch.norm(pred_points - target_points, dim=2).mean().item() * 1024  # 假设1024x1024
                test_metrics['pixel_error'] += pixel_error
                
                # 如果有地理坐标，计算地理误差
                if 'world_points' in targets:
                    # 获取变换矩阵
                    transform_matrix = self.model.get_transformation_matrix(outputs['transform_params'])
                    
                    # 计算变换后的控制点
                    warped_points = self.model.compute_warped_control_points(
                        outputs['control_points'],
                        transform_matrix
                    )
                    
                    # 计算地理坐标误差（以度为单位）
                    geo_error = torch.norm(warped_points - targets['world_points'], dim=2).mean().item()
                    test_metrics['geo_error'] += geo_error
                
                # 可视化每个批次的第一个样本
                self._visualize_test_results(
                    images[0],
                    pred_points[0],
                    target_points[0],
                    chart_paths[0],
                    metadata[0],
                    batch_idx
                )
                
                # 尝试保存校准结果到数据库
                for i, m in enumerate(metadata):
                    if 'id' in m:
                        chart_id = m['id']
                        try:
                            # 计算变换矩阵
                            transform_params = outputs['transform_params'][i].cpu().numpy()
                            if self.model.transformation_type == 'affine':
                                matrix = np.eye(3)
                                matrix[:2, :] = transform_params.reshape(2, 3)
                            else:  # perspective
                                matrix = np.ones((3, 3))
                                matrix.flat[:8] = transform_params
                            
                            # 控制点
                            control_points = outputs['control_points'][i].cpu().numpy() * 1024  # 转换回像素坐标
                            
                            # 计算准确度
                            if 'control_points' in targets:
                                target_control_points = targets['control_points'][i].cpu().numpy() * 1024
                                accuracy = 1.0 - np.mean(np.sqrt(np.sum((control_points - target_control_points)**2, axis=1))) / 100
                                accuracy = max(0.0, min(1.0, accuracy))  # 限制在[0,1]范围内
                            else:
                                accuracy = 0.5  # 默认中等准确度
                            
                            # 保存到数据库
                            self.db.save_calibration(
                                chart_id=chart_id,
                                transformation_matrix=matrix.tolist(),
                                reference_points=control_points.tolist(),
                                accuracy_score=float(accuracy)
                            )
                            
                            logger.info(f"已将校准结果保存到数据库，图表ID: {chart_id}")
                            
                        except Exception as e:
                            logger.error(f"保存校准结果失败: {e}")
        
        # 计算平均值
        test_loss /= num_batches
        for k in test_metrics:
            test_metrics[k] /= num_batches
        
        test_metrics['loss'] = test_loss
        
        # 打印测试结果
        logger.info(f"测试结果:")
        for k, v in test_metrics.items():
            logger.info(f"- {k}: {v:.4f}")
        
        return test_metrics
    
    def _visualize_test_results(self, 
                              image: torch.Tensor, 
                              pred_points: torch.Tensor, 
                              target_points: torch.Tensor,
                              chart_path: str,
                              metadata: Dict,
                              batch_idx: int):
        """
        可视化测试结果
        
        参数:
            image: 输入图像
            pred_points: 预测的控制点 [num_points, 2]
            target_points: 目标控制点 [num_points, 2]
            chart_path: 图表路径
            metadata: 元数据
            batch_idx: 批次索引
        """
        # 转换为numpy数组
        img = image.cpu().permute(1, 2, 0).numpy()
        
        # 反归一化图像
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img)
        
        height, width = img.shape[:2]
        
        # 绘制预测点
        pred_points_np = pred_points.cpu().numpy()
        ax.scatter(
            pred_points_np[:, 0] * width, 
            pred_points_np[:, 1] * height, 
            c='r', 
            marker='x', 
            s=100, 
            label='预测点'
        )
        
        # 绘制目标点
        target_points_np = target_points.cpu().numpy()
        ax.scatter(
            target_points_np[:, 0] * width, 
            target_points_np[:, 1] * height, 
            c='g', 
            marker='o', 
            s=100, 
            label='目标点'
        )
        
        # 添加图表信息
        chart_name = Path(chart_path).stem
        ax.set_title(f"图表: {chart_name}")
        
        # 添加误差信息
        pixel_error = np.mean(np.sqrt(np.sum((pred_points_np - target_points_np)**2, axis=1))) * width
        ax.text(
            0.02, 0.98, 
            f"平均像素误差: {pixel_error:.2f}px", 
            transform=ax.transAxes, 
            fontsize=12,
            verticalalignment='top'
        )
        
        # 如果有地理坐标，添加地理误差信息
        if 'calibration' in metadata:
            ax.text(
                0.02, 0.94, 
                f"机场: {metadata.get('airport_icao', 'Unknown')}", 
                transform=ax.transAxes, 
                fontsize=12,
                verticalalignment='top'
            )
        
        ax.legend()
        
        # 保存到文件
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        output_path = output_dir / f"{chart_name}_test_{batch_idx}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        
        # 保存到TensorBoard
        self.writer.add_figure(
            f'test/results_{batch_idx}', 
            fig, 
            global_step=self.current_epoch
        )
        
        plt.close()
