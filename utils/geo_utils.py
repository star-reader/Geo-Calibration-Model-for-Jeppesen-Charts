import numpy as np
import math
from typing import List, Tuple, Dict, Union, Optional
import pyproj
from scipy.optimize import least_squares

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点间的大圆距离
    
    参数:
        lat1: 第一点纬度（度）
        lon1: 第一点经度（度）
        lat2: 第二点纬度（度）
        lon2: 第二点经度（度）
        
    返回:
        距离（海里）
    """
    # 将十进制度转换为弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3440.065  # 地球半径（海里）
    
    return c * r

def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点之间的初始方位角
    
    参数:
        lat1: 第一点纬度（度）
        lon1: 第一点经度（度）
        lat2: 第二点纬度（度）
        lon2: 第二点经度（度）
        
    返回:
        方位角（度，从北顺时针）
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    y = np.sin(lon2-lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2-lon1)
    bearing = np.arctan2(y, x)
    
    # 转换为度数并确保在0-360范围内
    bearing_deg = (np.degrees(bearing) + 360) % 360
    
    return bearing_deg

def destination_point(lat: float, lon: float, bearing: float, distance: float) -> Tuple[float, float]:
    """
    给定起点、方位角和距离，计算终点坐标
    
    参数:
        lat: 起点纬度（度）
        lon: 起点经度（度）
        bearing: 方位角（度，从北顺时针）
        distance: 距离（海里）
        
    返回:
        终点坐标(lat, lon)
    """
    # 转换为弧度
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bearing_rad = np.radians(bearing)
    
    # 地球半径（海里）
    R = 3440.065
    
    # 角距离
    d_R = distance / R
    
    # 计算终点纬度
    lat2_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(d_R) + 
        np.cos(lat_rad) * np.sin(d_R) * np.cos(bearing_rad)
    )
    
    # 计算终点经度
    lon2_rad = lon_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(d_R) * np.cos(lat_rad),
        np.cos(d_R) - np.sin(lat_rad) * np.sin(lat2_rad)
    )
    
    # 转换回度数
    lat2 = np.degrees(lat2_rad)
    lon2 = np.degrees(lon2_rad)
    
    return lat2, lon2

def create_transformation_matrix(src_points: np.ndarray, dst_points: np.ndarray, 
                               transform_type: str = 'perspective') -> np.ndarray:
    """
    创建从源点到目标点的变换矩阵
    
    参数:
        src_points: 源点坐标，形状为 (n, 2)
        dst_points: 目标点坐标，形状为 (n, 2)
        transform_type: 变换类型，'affine' 或 'perspective'
        
    返回:
        变换矩阵
    """
    if transform_type == 'affine':
        # 仿射变换：最少需要3个点
        if src_points.shape[0] < 3 or dst_points.shape[0] < 3:
            raise ValueError("仿射变换至少需要3个点")
        
        # 使用OpenCV计算仿射变换矩阵
        import cv2
        matrix = cv2.getAffineTransform(
            src_points[:3].astype(np.float32), 
            dst_points[:3].astype(np.float32)
        )
        
        # 转换为3x3矩阵
        affine_matrix = np.eye(3, 3)
        affine_matrix[:2, :] = matrix
        return affine_matrix
        
    elif transform_type == 'perspective':
        # 透视变换：最少需要4个点
        if src_points.shape[0] < 4 or dst_points.shape[0] < 4:
            raise ValueError("透视变换至少需要4个点")
        
        # 使用OpenCV计算透视变换矩阵
        import cv2
        matrix = cv2.getPerspectiveTransform(
            src_points[:4].astype(np.float32), 
            dst_points[:4].astype(np.float32)
        )
        return matrix
    
    else:
        raise ValueError(f"不支持的变换类型: {transform_type}")

def apply_transformation(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    应用变换矩阵到点集
    
    参数:
        points: 输入点坐标，形状为 (n, 2)
        matrix: 3x3变换矩阵
        
    返回:
        变换后的点坐标，形状为 (n, 2)
    """
    # 确保输入是numpy数组
    points = np.asarray(points)
    
    # 转换为齐次坐标
    n = points.shape[0]
    homogeneous = np.ones((n, 3))
    homogeneous[:, :2] = points
    
    # 应用变换
    transformed = np.dot(homogeneous, matrix.T)
    
    # 从齐次坐标转回
    result = transformed[:, :2] / transformed[:, 2:3]
    
    return result

def optimize_transformation(src_points: np.ndarray, dst_points: np.ndarray, 
                          initial_matrix: np.ndarray = None,
                          transform_type: str = 'perspective') -> np.ndarray:
    """
    优化变换矩阵以最小化源点到目标点的映射误差
    
    参数:
        src_points: 源点坐标，形状为 (n, 2)
        dst_points: 目标点坐标，形状为 (n, 2)
        initial_matrix: 初始变换矩阵（可选）
        transform_type: 变换类型，'affine' 或 'perspective'
        
    返回:
        优化后的变换矩阵
    """
    if initial_matrix is None:
        # 创建初始变换矩阵
        initial_matrix = create_transformation_matrix(src_points, dst_points, transform_type)
    
    # 提取变换参数
    if transform_type == 'perspective':
        # 8参数（3x3矩阵，最后一个元素固定为1）
        initial_params = initial_matrix.flatten()[:8]
        
        def objective(params):
            matrix = np.ones((3, 3))
            matrix.flat[:8] = params
            transformed = apply_transformation(src_points, matrix)
            return (transformed - dst_points).flatten()
        
    elif transform_type == 'affine':
        # 6参数（2x3矩阵）
        initial_params = initial_matrix[:2, :].flatten()
        
        def objective(params):
            matrix = np.eye(3)
            matrix[:2, :] = params.reshape(2, 3)
            transformed = apply_transformation(src_points, matrix)
            return (transformed - dst_points).flatten()
    
    else:
        raise ValueError(f"不支持的变换类型: {transform_type}")
    
    # 优化
    result = least_squares(objective, initial_params, method='lm')
    optimized_params = result.x
    
    # 创建最终矩阵
    if transform_type == 'perspective':
        optimized_matrix = np.ones((3, 3))
        optimized_matrix.flat[:8] = optimized_params
    else:
        optimized_matrix = np.eye(3)
        optimized_matrix[:2, :] = optimized_params.reshape(2, 3)
    
    return optimized_matrix

def pixel_to_geo(pixel_x: float, pixel_y: float, matrix: np.ndarray) -> Tuple[float, float]:
    """
    将像素坐标转换为地理坐标
    
    参数:
        pixel_x: 像素X坐标
        pixel_y: 像素Y坐标
        matrix: 像素到地理坐标的变换矩阵
        
    返回:
        地理坐标 (lat, lon)
    """
    points = np.array([[pixel_x, pixel_y]])
    geo = apply_transformation(points, matrix)[0]
    return geo[1], geo[0]  # 返回 (lat, lon)

def geo_to_pixel(lat: float, lon: float, matrix: np.ndarray) -> Tuple[float, float]:
    """
    将地理坐标转换为像素坐标
    
    参数:
        lat: 纬度
        lon: 经度
        matrix: 地理坐标到像素的变换矩阵（像素到地理的逆矩阵）
        
    返回:
        像素坐标 (x, y)
    """
    # 计算逆矩阵
    inv_matrix = np.linalg.inv(matrix)
    
    # 应用变换
    points = np.array([[lon, lat]])  # 注意顺序是 (lon, lat)
    pixel = apply_transformation(points, inv_matrix)[0]
    
    return pixel[0], pixel[1]

def calculate_transformation_accuracy(src_points: np.ndarray, dst_points: np.ndarray, 
                                    matrix: np.ndarray) -> Dict[str, float]:
    """
    计算变换矩阵的精度
    
    参数:
        src_points: 源点坐标，形状为 (n, 2)
        dst_points: 目标点坐标，形状为 (n, 2)
        matrix: 变换矩阵
        
    返回:
        包含不同精度指标的字典
    """
    # 变换源点
    transformed = apply_transformation(src_points, matrix)
    
    # 计算误差
    errors = dst_points - transformed
    
    # 计算欧氏距离
    distances = np.sqrt(np.sum(errors**2, axis=1))
    
    # 计算均方根误差
    rmse = np.sqrt(np.mean(distances**2))
    
    # 计算平均绝对误差
    mae = np.mean(distances)
    
    # 计算最大误差
    max_error = np.max(distances)
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'max_error': float(max_error),
        'individual_errors': distances.tolist()
    }

def coordinates_to_mercator(lat: float, lon: float) -> Tuple[float, float]:
    """
    将地理坐标转换为墨卡托投影坐标
    
    参数:
        lat: 纬度（度）
        lon: 经度（度）
        
    返回:
        墨卡托坐标 (x, y)
    """
    # 使用pyproj进行投影变换
    p = pyproj.Proj(proj='merc', datum='WGS84')
    x, y = p(lon, lat)
    return x, y

def mercator_to_coordinates(x: float, y: float) -> Tuple[float, float]:
    """
    将墨卡托投影坐标转换为地理坐标
    
    参数:
        x: 墨卡托X坐标
        y: 墨卡托Y坐标
        
    返回:
        地理坐标 (lat, lon)
    """
    # 使用pyproj进行投影逆变换
    p = pyproj.Proj(proj='merc', datum='WGS84')
    lon, lat = p(x, y, inverse=True)
    return lat, lon

def calculate_scale(matrix: np.ndarray, lat: float) -> float:
    """
    计算变换矩阵在给定纬度处的近似比例尺
    
    参数:
        matrix: 变换矩阵
        lat: 参考纬度
        
    返回:
        比例尺（像素/度）
    """
    # 在给定纬度处选择两个相距1度经度的点
    lon1, lon2 = 0, 1
    
    # 转换为像素坐标
    px1, py1 = geo_to_pixel(lat, lon1, matrix)
    px2, py2 = geo_to_pixel(lat, lon2, matrix)
    
    # 计算像素距离
    pixel_distance = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)
    
    # 计算物理距离（1度经度在该纬度处的距离，单位：海里）
    physical_distance = haversine_distance(lat, lon1, lat, lon2)
    
    # 返回比例尺（像素/海里）
    return pixel_distance / physical_distance if physical_distance > 0 else 0
