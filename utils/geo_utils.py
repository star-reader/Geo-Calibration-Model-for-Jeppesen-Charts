import numpy as np
import math
from typing import List, Tuple, Dict, Union, Optional
import pyproj
from scipy.optimize import least_squares

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # 将十进制度转换为弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3440.065
    
    return c * r

def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    y = np.sin(lon2-lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2-lon1)
    bearing = np.arctan2(y, x)
    
    # 转换为度数并确保在0-360范围内
    bearing_deg = (np.degrees(bearing) + 360) % 360
    
    return bearing_deg

def destination_point(lat: float, lon: float, bearing: float, distance: float) -> Tuple[float, float]:
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bearing_rad = np.radians(bearing)
    
    R = 3440.065
    d_R = distance / R
    lat2_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(d_R) + 
        np.cos(lat_rad) * np.sin(d_R) * np.cos(bearing_rad)
    )
    lon2_rad = lon_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(d_R) * np.cos(lat_rad),
        np.cos(d_R) - np.sin(lat_rad) * np.sin(lat2_rad)
    )
    
    lat2 = np.degrees(lat2_rad)
    lon2 = np.degrees(lon2_rad)
    
    return lat2, lon2

def create_transformation_matrix(src_points: np.ndarray, dst_points: np.ndarray, 
                               transform_type: str = 'perspective') -> np.ndarray:

    if transform_type == 'affine':
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
        if src_points.shape[0] < 4 or dst_points.shape[0] < 4:
            raise ValueError("透视变换至少需要4个点")
        
        import cv2
        matrix = cv2.getPerspectiveTransform(
            src_points[:4].astype(np.float32), 
            dst_points[:4].astype(np.float32)
        )
        return matrix
    
    else:
        raise ValueError(f"不支持的变换类型: {transform_type}")

def apply_transformation(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    
    n = points.shape[0]
    homogeneous = np.ones((n, 3))
    homogeneous[:, :2] = points
    
    transformed = np.dot(homogeneous, matrix.T)
    result = transformed[:, :2] / transformed[:, 2:3]
    
    return result

def optimize_transformation(src_points: np.ndarray, dst_points: np.ndarray, 
                          initial_matrix: np.ndarray = None,
                          transform_type: str = 'perspective') -> np.ndarray:
    if initial_matrix is None:
        initial_matrix = create_transformation_matrix(src_points, dst_points, transform_type)
    
    if transform_type == 'perspective':
        # 8参数（3x3矩阵，最后一个元素固定为1）
        initial_params = initial_matrix.flatten()[:8]
        
        def objective(params):
            matrix = np.ones((3, 3))
            matrix.flat[:8] = params
            transformed = apply_transformation(src_points, matrix)
            return (transformed - dst_points).flatten()
        
    elif transform_type == 'affine':
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
    
    if transform_type == 'perspective':
        optimized_matrix = np.ones((3, 3))
        optimized_matrix.flat[:8] = optimized_params
    else:
        optimized_matrix = np.eye(3)
        optimized_matrix[:2, :] = optimized_params.reshape(2, 3)
    
    return optimized_matrix

def pixel_to_geo(pixel_x: float, pixel_y: float, matrix: np.ndarray) -> Tuple[float, float]:
    points = np.array([[pixel_x, pixel_y]])
    geo = apply_transformation(points, matrix)[0]
    return geo[1], geo[0]  # 返回 (lat, lon)

def geo_to_pixel(lat: float, lon: float, matrix: np.ndarray) -> Tuple[float, float]:
    inv_matrix = np.linalg.inv(matrix)

    points = np.array([[lon, lat]])  # 注意顺序是 (lon, lat)
    pixel = apply_transformation(points, inv_matrix)[0]
    
    return pixel[0], pixel[1]

def calculate_transformation_accuracy(src_points: np.ndarray, dst_points: np.ndarray, 
                                    matrix: np.ndarray) -> Dict[str, float]:
    transformed = apply_transformation(src_points, matrix)
    errors = dst_points - transformed
    distances = np.sqrt(np.sum(errors**2, axis=1))
    rmse = np.sqrt(np.mean(distances**2))
    

    mae = np.mean(distances)
    
    max_error = np.max(distances)
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'max_error': float(max_error),
        'individual_errors': distances.tolist()
    }

def coordinates_to_mercator(lat: float, lon: float) -> Tuple[float, float]:
    p = pyproj.Proj(proj='merc', datum='WGS84')
    x, y = p(lon, lat)
    return x, y

def mercator_to_coordinates(x: float, y: float) -> Tuple[float, float]:
    p = pyproj.Proj(proj='merc', datum='WGS84')
    lon, lat = p(x, y, inverse=True)
    return lat, lon

def calculate_scale(matrix: np.ndarray, lat: float) -> float:
    lon1, lon2 = 0, 1
    
    # 转换为像素坐标
    px1, py1 = geo_to_pixel(lat, lon1, matrix)
    px2, py2 = geo_to_pixel(lat, lon2, matrix)
    
    pixel_distance = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)
    
    # 计算物理距离（1度经度在该纬度处的距离，单位：海里）
    physical_distance = haversine_distance(lat, lon1, lat, lon2)
    return pixel_distance / physical_distance if physical_distance > 0 else 0
