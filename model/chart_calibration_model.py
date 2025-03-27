import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and ReLU activation."""
    
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
    """Residual block with skip connection."""
    
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
    """Spatial attention module."""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate spatial attention weights
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.conv(pool))
        
        # Apply attention weights
        return x * attention


class FeatureFusionModule(nn.Module):
    """Feature fusion module to combine multi-scale features."""
    
    def __init__(self, channels: List[int]):
        super(FeatureFusionModule, self).__init__()
        self.channels = channels
        
        # Convolutional layers to transform each input to the same channel size
        self.transforms = nn.ModuleList([
            nn.Conv2d(c, channels[-1], kernel_size=1)
            for c in channels
        ])
        
        # 1x1 convolution after concatenation
        self.fusion_conv = nn.Conv2d(channels[-1] * len(channels), channels[-1], kernel_size=1)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Transform all features to the same channel dimensions
        transformed = []
        for i, feature in enumerate(features):
            # Resize to match the size of the last feature
            if feature.size(2) != features[-1].size(2) or feature.size(3) != features[-1].size(3):
                feature = F.interpolate(
                    feature, 
                    size=(features[-1].size(2), features[-1].size(3)), 
                    mode='bilinear', 
                    align_corners=False
                )
            transformed.append(self.transforms[i](feature))
        
        # Concatenate all features
        concat = torch.cat(transformed, dim=1)
        
        # Apply fusion convolution
        fused = self.fusion_conv(concat)
        
        # Apply spatial attention
        attended = self.spatial_attention(fused)
        
        return attended


class ControlPointRegressor(nn.Module):
    """Regressor for predicting control point coordinates."""
    
    def __init__(self, in_channels: int, num_points: int):
        super(ControlPointRegressor, self).__init__()
        self.num_points = num_points
        
        self.conv1 = ConvBlock(in_channels, in_channels // 2)
        self.conv2 = ConvBlock(in_channels // 2, in_channels // 4)
        
        # Global average pooling + FC layers for coordinate regression
        self.fc1 = nn.Linear(in_channels // 4, 256)
        self.fc2 = nn.Linear(256, num_points * 2)  # x, y coordinates for each point
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Reshape to (batch_size, num_points, 2)
        return x.view(-1, self.num_points, 2)


class TransformationMatrixRegressor(nn.Module):
    """Regressor for predicting transformation matrix parameters."""
    
    def __init__(self, in_channels: int, matrix_type: str = 'affine'):
        super(TransformationMatrixRegressor, self).__init__()
        self.matrix_type = matrix_type
        
        # Number of parameters to predict
        if matrix_type == 'affine':
            self.num_params = 6  # 2x3 affine matrix
        elif matrix_type == 'perspective':
            self.num_params = 8  # 3x3 perspective matrix (last element is fixed at 1)
        else:
            raise ValueError(f"Unsupported matrix type: {matrix_type}")
        
        self.conv1 = ConvBlock(in_channels, in_channels // 2)
        self.conv2 = ConvBlock(in_channels // 2, in_channels // 4)
        
        # Global average pooling + FC layers for matrix regression
        self.fc1 = nn.Linear(in_channels // 4, 256)
        self.fc2 = nn.Linear(256, self.num_params)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # For affine transformation, initialize with identity-like transform
        if self.matrix_type == 'affine':
            # Add identity transform bias
            identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32).to(x.device)
            x = x + identity
        elif self.matrix_type == 'perspective':
            # Add identity transform bias
            identity = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32).to(x.device)
            x = x + identity
        
        return x


class ChartCalibrationModel(nn.Module):
    """
    Deep learning model for calibrating Jeppesen charts.
    Uses a CNN backbone to extract features, followed by specialized heads for different tasks.
    """
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 pretrained: bool = True,
                 num_control_points: int = 4,
                 transformation_type: str = 'perspective'):
        """
        Initialize the chart calibration model.
        
        Args:
            backbone: Backbone CNN architecture
            pretrained: Whether to use pretrained weights
            num_control_points: Number of control points to predict
            transformation_type: Type of transformation matrix
        """
        super(ChartCalibrationModel, self).__init__()
        self.num_control_points = num_control_points
        self.transformation_type = transformation_type
        
        # Load backbone CNN
        if backbone == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            self.layer0 = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool
            )
            self.layer1 = base_model.layer1  # 256 channels
            self.layer2 = base_model.layer2  # 512 channels
            self.layer3 = base_model.layer3  # 1024 channels
            self.layer4 = base_model.layer4  # 2048 channels
            
            # Feature channels at each layer
            feature_channels = [256, 512, 1024, 2048]
            
        elif backbone == 'efficientnet_b3':
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b3(weights=weights)
            
            # Decompose EfficientNet features into layers
            self.layer0 = nn.Sequential(
                base_model.features[0],
                base_model.features[1]
            )
            self.layer1 = nn.Sequential(*base_model.features[2:4])  # 40 channels
            self.layer2 = nn.Sequential(*base_model.features[4:6])  # 80 channels
            self.layer3 = nn.Sequential(*base_model.features[6:8])  # 192 channels
            self.layer4 = nn.Sequential(*base_model.features[8:])   # 1536 channels
            
            # Feature channels at each layer
            feature_channels = [40, 80, 192, 1536]
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature fusion module
        self.fusion = FeatureFusionModule(feature_channels)
        
        # Control points head
        self.control_points_head = ControlPointRegressor(
            in_channels=feature_channels[-1],
            num_points=num_control_points
        )
        
        # Transformation matrix head
        self.transform_head = TransformationMatrixRegressor(
            in_channels=feature_channels[-1],
            matrix_type=transformation_type
        )
        
        # Extra residual blocks for refinement
        self.refine1 = ResidualBlock(feature_channels[-1])
        self.refine2 = ResidualBlock(feature_channels[-1])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Dictionary with predicted outputs
        """
        # Extract features from backbone
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Fuse multi-scale features
        fused = self.fusion([x1, x2, x3, x4])
        
        # Refine features
        refined = self.refine1(fused)
        refined = self.refine2(refined)
        
        # Predict control points
        control_points = self.control_points_head(refined)
        
        # Predict transformation matrix
        transform_params = self.transform_head(refined)
        
        # Normalize control points to [0, 1] range
        control_points_norm = torch.sigmoid(control_points)
        
        return {
            'control_points': control_points_norm,
            'transform_params': transform_params
        }
    
    def get_transformation_matrix(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert the predicted parameters to a transformation matrix.
        
        Args:
            params: Predicted transformation parameters
            
        Returns:
            Transformation matrix tensor
        """
        batch_size = params.size(0)
        
        if self.transformation_type == 'affine':
            # Create 2x3 affine matrix
            matrix = params.view(batch_size, 2, 3)
            
            # Append row of [0, 0, 1] to make it 3x3
            zeros = torch.zeros(batch_size, 1, 3, device=params.device)
            zeros[:, :, 2] = 1
            matrix = torch.cat([matrix, zeros], dim=1)
            
        elif self.transformation_type == 'perspective':
            # Create 3x3 perspective matrix with last element fixed at 1
            matrix = torch.ones(batch_size, 3, 3, device=params.device)
            matrix[:, 0:2, 0:3] = params[:, 0:6].view(batch_size, 2, 3)
            matrix[:, 2, 0:2] = params[:, 6:8]
            
        return matrix
    
    def compute_warped_control_points(self, control_points: torch.Tensor, 
                                     transform_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute control points warped by the transformation matrix.
        
        Args:
            control_points: Control points of shape (batch_size, num_points, 2)
            transform_matrix: Transformation matrix of shape (batch_size, 3, 3)
            
        Returns:
            Warped control points of shape (batch_size, num_points, 2)
        """
        batch_size = control_points.size(0)
        num_points = control_points.size(1)
        
        # Reshape control points to homogeneous coordinates
        ones = torch.ones(batch_size, num_points, 1, device=control_points.device)
        points_h = torch.cat([control_points, ones], dim=2)  # (batch_size, num_points, 3)
        
        # Transpose for matrix multiplication
        points_h = points_h.transpose(1, 2)  # (batch_size, 3, num_points)
        
        # Apply transformation
        warped_h = torch.bmm(transform_matrix, points_h)  # (batch_size, 3, num_points)
        
        # Normalize by homogeneous coordinate
        warped = warped_h[:, 0:2, :] / (warped_h[:, 2:3, :] + 1e-8)
        
        # Transpose back to original shape
        warped = warped.transpose(1, 2)  # (batch_size, num_points, 2)
        
        return warped
    
    def convert_to_image_coordinates(self, points_norm: torch.Tensor, 
                                    height: int, width: int) -> torch.Tensor:
        """
        Convert normalized control points [0, 1] to image coordinates.
        
        Args:
            points_norm: Normalized points of shape (batch_size, num_points, 2)
            height: Image height
            width: Image width
            
        Returns:
            Points in image coordinates of shape (batch_size, num_points, 2)
        """
        batch_size = points_norm.size(0)
        num_points = points_norm.size(1)
        
        # Scale to image dimensions
        points_img = torch.zeros_like(points_norm)
        points_img[:, :, 0] = points_norm[:, :, 0] * width
        points_img[:, :, 1] = points_norm[:, :, 1] * height
        
        return points_img


class ChartCalibrationLoss(nn.Module):
    """
    Combined loss function for chart calibration task.
    Includes terms for control point accuracy and transformation matrix consistency.
    """
    
    def __init__(self, 
                 point_weight: float = 1.0,
                 transform_weight: float = 1.0,
                 consistency_weight: float = 0.5,
                 smoothness_weight: float = 0.1):
        """
        Initialize the loss function.
        
        Args:
            point_weight: Weight for control point loss
            transform_weight: Weight for transformation matrix loss
            consistency_weight: Weight for consistency loss
            smoothness_weight: Weight for smoothness regularization
        """
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
        """
        Compute the combined loss.
        
        Args:
            predictions: Dictionary of predicted outputs
            targets: Dictionary of target values
            
        Returns:
            Dictionary of loss components and total loss
        """
        # Extract predictions
        pred_points = predictions['control_points']
        pred_transform = predictions['transform_params']
        
        # Extract targets
        target_points = targets['control_points']
        target_transform = targets.get('transform_params')
        
        # Control point loss
        point_loss = self.smooth_l1_loss(pred_points, target_points)
        
        # Initialize other losses
        transform_loss = torch.tensor(0.0, device=pred_points.device)
        consistency_loss = torch.tensor(0.0, device=pred_points.device)
        smoothness_loss = torch.tensor(0.0, device=pred_points.device)
        
        # Transformation matrix loss (if target available)
        if target_transform is not None:
            transform_loss = self.smooth_l1_loss(pred_transform, target_transform)
        
        # Compute model
        model = predictions.get('model')
        
        # Consistency loss: check if warped control points match target points in world coordinates
        if model is not None and 'world_points' in targets:
            warped_points = model.compute_warped_control_points(
                pred_points, 
                model.get_transformation_matrix(pred_transform)
            )
            target_world = targets['world_points']
            consistency_loss = self.mse_loss(warped_points, target_world)
        
        # Smoothness regularization for transformation parameters
        smoothness_loss = torch.norm(pred_transform, p=2)
        
        # Combine losses
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
    """
    Factory function to create chart calibration model.
    
    Args:
        backbone: CNN backbone architecture
        pretrained: Whether to use pretrained weights
        num_control_points: Number of control points to predict
        transformation_type: Type of transformation matrix
        
    Returns:
        Initialized chart calibration model
    """
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
    """
    Factory function to create chart calibration loss.
    
    Args:
        point_weight: Weight for control point loss
        transform_weight: Weight for transformation matrix loss
        consistency_weight: Weight for consistency loss
        smoothness_weight: Weight for smoothness regularization
        
    Returns:
        Initialized chart calibration loss function
    """
    loss_fn = ChartCalibrationLoss(
        point_weight=point_weight,
        transform_weight=transform_weight,
        consistency_weight=consistency_weight,
        smoothness_weight=smoothness_weight
    )
    
    return loss_fn
