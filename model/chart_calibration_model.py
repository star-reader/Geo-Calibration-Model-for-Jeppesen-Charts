import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, use_bn: bool = True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return F.relu(x)


class SpatialAttention(nn.Module):
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 算一下spatial attention weights
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.conv(pool))
        
        return x * attention


class FeatureFusionModule(nn.Module):
    """Feature fusion module to combine multi-scale features."""
    
    def __init__(self, channels: List[int]):
        super(FeatureFusionModule, self).__init__()
        self.channels = channels

        self.transforms = nn.ModuleList([
            nn.Conv2d(c, channels[-1], kernel_size=1)
            for c in channels
        ])
        
        # 1x1
        self.fusion_conv = nn.Conv2d(channels[-1] * len(channels), channels[-1], kernel_size=1)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        transformed = []
        for i, feature in enumerate(features):
            if feature.size(2) != features[-1].size(2) or feature.size(3) != features[-1].size(3):
                feature = F.interpolate(
                    feature, 
                    size=(features[-1].size(2), features[-1].size(3)), 
                    mode='bilinear', 
                    align_corners=False
                )
            transformed.append(self.transforms[i](feature))
        
        # 转换开始
        concat = torch.cat(transformed, dim=1)
        fused = self.fusion_conv(concat)
        attended = self.spatial_attention(fused)
        
        return attended


class ControlPointRegressor(nn.Module):
    def __init__(self, in_channels: int, num_points: int):
        super(ControlPointRegressor, self).__init__()
        self.num_points = num_points
        
        self.conv1 = ConvBlock(in_channels, in_channels // 2)
        self.conv2 = ConvBlock(in_channels // 2, in_channels // 4)
        
        self.fc1 = nn.Linear(in_channels // 4, 256)
        self.fc2 = nn.Linear(256, num_points * 2)  # x, y coordinates
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.view(-1, self.num_points, 2)


class TransformationMatrixRegressor(nn.Module):
    def __init__(self, in_channels: int, matrix_type: str = 'affine'):
        super(TransformationMatrixRegressor, self).__init__()
        self.matrix_type = matrix_type
        
        if matrix_type == 'affine':
            self.num_params = 6  # 2x3
        elif matrix_type == 'perspective':
            self.num_params = 8  # 3x3
        else:
            raise ValueError(f"Unsupported matrix type: {matrix_type}")
        
        self.conv1 = ConvBlock(in_channels, in_channels // 2)
        self.conv2 = ConvBlock(in_channels // 2, in_channels // 4)
        
        self.fc1 = nn.Linear(in_channels // 4, 256)
        self.fc2 = nn.Linear(256, self.num_params)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        if self.matrix_type == 'affine':
            identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32).to(x.device)
            x = x + identity
        elif self.matrix_type == 'perspective':
            identity = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32).to(x.device)
            x = x + identity
        
        return x


class ChartCalibrationModel(nn.Module):
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 pretrained: bool = True,
                 num_control_points: int = 4,
                 transformation_type: str = 'perspective'):
        
        super(ChartCalibrationModel, self).__init__()
        self.num_control_points = num_control_points
        self.transformation_type = transformation_type
        
        # 加载CNN
        if backbone == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            self.layer0 = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool
            )
            self.layer1 = base_model.layer1  # 256
            self.layer2 = base_model.layer2  # 512
            self.layer3 = base_model.layer3  # 1024
            self.layer4 = base_model.layer4  # 2048
            
            feature_channels = [256, 512, 1024, 2048]
            
        elif backbone == 'efficientnet_b3':
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b3(weights=weights)
            
            self.layer0 = nn.Sequential(
                base_model.features[0],
                base_model.features[1]
            )
            self.layer1 = nn.Sequential(*base_model.features[2:4])
            self.layer2 = nn.Sequential(*base_model.features[4:6])
            self.layer3 = nn.Sequential(*base_model.features[6:8])
            self.layer4 = nn.Sequential(*base_model.features[8:])
            
            feature_channels = [40, 80, 192, 1536]
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.fusion = FeatureFusionModule(feature_channels)
        
        self.control_points_head = ControlPointRegressor(
            in_channels=feature_channels[-1],
            num_points=num_control_points
        )
        self.transform_head = TransformationMatrixRegressor(
            in_channels=feature_channels[-1],
            matrix_type=transformation_type
        )
        
        self.refine1 = ResidualBlock(feature_channels[-1])
        self.refine2 = ResidualBlock(feature_channels[-1])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        fused = self.fusion([x1, x2, x3, x4])
        
        # Refine features
        refined = self.refine1(fused)
        refined = self.refine2(refined)
        
        control_points = self.control_points_head(refined)
        
        transform_params = self.transform_head(refined)
        
        # [0, 1] range
        control_points_norm = torch.sigmoid(control_points)
        
        return {
            'control_points': control_points_norm,
            'transform_params': transform_params
        }
    
    def get_transformation_matrix(self, params: torch.Tensor) -> torch.Tensor:
        batch_size = params.size(0)
        
        if self.transformation_type == 'affine':
            matrix = params.view(batch_size, 2, 3)

            zeros = torch.zeros(batch_size, 1, 3, device=params.device)
            zeros[:, :, 2] = 1
            matrix = torch.cat([matrix, zeros], dim=1)
            
        elif self.transformation_type == 'perspective':
            matrix = torch.ones(batch_size, 3, 3, device=params.device)
            matrix[:, 0:2, 0:3] = params[:, 0:6].view(batch_size, 2, 3)
            matrix[:, 2, 0:2] = params[:, 6:8]
            
        return matrix
    
    def compute_warped_control_points(self, control_points: torch.Tensor, 
                                     transform_matrix: torch.Tensor) -> torch.Tensor:
        batch_size = control_points.size(0)
        num_points = control_points.size(1)
        
        ones = torch.ones(batch_size, num_points, 1, device=control_points.device)
        points_h = torch.cat([control_points, ones], dim=2)
        points_h = points_h.transpose(1, 2)
        warped_h = torch.bmm(transform_matrix, points_h)  # (batch_size, 3, num_points)
        warped = warped_h[:, 0:2, :] / (warped_h[:, 2:3, :] + 1e-8)
    
        warped = warped.transpose(1, 2)
        
        return warped
    
    def convert_to_image_coordinates(self, points_norm: torch.Tensor, 
                                    height: int, width: int) -> torch.Tensor:
        batch_size = points_norm.size(0)
        num_points = points_norm.size(1)
        
        # Scale to image dimensions
        points_img = torch.zeros_like(points_norm)
        points_img[:, :, 0] = points_norm[:, :, 0] * width
        points_img[:, :, 1] = points_norm[:, :, 1] * height
        
        return points_img


class ChartCalibrationLoss(nn.Module):
    def __init__(self, 
                 point_weight: float = 1.0,
                 transform_weight: float = 1.0,
                 consistency_weight: float = 0.5,
                 smoothness_weight: float = 0.1):
        super(ChartCalibrationLoss, self).__init__()
        self.point_weight = point_weight
        self.transform_weight = transform_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
    
    def forward(self, 
               predictions: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred_points = predictions['control_points']
        pred_transform = predictions['transform_params']
        
        target_points = targets['control_points']
        target_transform = targets.get('transform_params')

        point_loss = self.smooth_l1_loss(pred_points, target_points)
        
        # 初始化其他的losses
        transform_loss = torch.tensor(0.0, device=pred_points.device)
        consistency_loss = torch.tensor(0.0, device=pred_points.device)
        smoothness_loss = torch.tensor(0.0, device=pred_points.device)
        
        if target_transform is not None:
            transform_loss = self.smooth_l1_loss(pred_transform, target_transform)
        
        model = predictions.get('model')
        if model is not None and 'world_points' in targets:
            warped_points = model.compute_warped_control_points(
                pred_points, 
                model.get_transformation_matrix(pred_transform)
            )
            target_world = targets['world_points']
            consistency_loss = self.mse_loss(warped_points, target_world)
        
        smoothness_loss = torch.norm(pred_transform, p=2)
        total_loss = (
            self.point_weight * point_loss +
            self.transform_weight * transform_loss +
            self.consistency_weight * consistency_loss +
            self.smoothness_weight * smoothness_loss
        )
        
        return {
            'total_loss': total_loss,
            'point_loss': point_loss,
            'transform_loss': transform_loss,
            'consistency_loss': consistency_loss,
            'smoothness_loss': smoothness_loss
        }


def create_chart_calibration_model(
    backbone: str = 'resnet50',
    pretrained: bool = True,
    num_control_points: int = 4,
    transformation_type: str = 'perspective'
) -> ChartCalibrationModel:
    model = ChartCalibrationModel(
        backbone=backbone,
        pretrained=pretrained,
        num_control_points=num_control_points,
        transformation_type=transformation_type
    )
    
    return model


def create_chart_calibration_loss(
    point_weight: float = 1.0,
    transform_weight: float = 1.0,
    consistency_weight: float = 0.5,
    smoothness_weight: float = 0.1
) -> ChartCalibrationLoss:
    loss_fn = ChartCalibrationLoss(
        point_weight=point_weight,
        transform_weight=transform_weight,
        consistency_weight=consistency_weight,
        smoothness_weight=smoothness_weight
    )
    
    return loss_fn
